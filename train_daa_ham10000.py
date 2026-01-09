#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Archetypal Analysis on HAM10000.

Key improvements vs the JAFFE script you shared:
- One CLI argument --data-dir points to the *root* HAM10000 folder (with both image parts + metadata csv).
- Optional filtering by dx and/or localization (and more).
- Group split by lesion_id (avoid leakage across train/test).
- Streaming input pipeline (no need to load all 10k images into RAM).
- Results path bug fixed (uses --results-path).
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
from datetime import datetime
from itertools import compress
from pathlib import Path
import math

import numpy as np
import pandas as pd

# Headless-friendly matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except Exception:  # pragma: no cover
    import tensorflow as tf  # type: ignore

# custom libs
from AT_lib import lib_vae, lib_at, lib_plt
from datasets import ham10000

# distribution lib is in lib_vae (tfd); for kl_divergence we use tf.distributions / tfp via that module
try:  # pragma: no cover
    import tensorflow_probability as tfp  # type: ignore
    tfd = tfp.distributions
except Exception:  # pragma: no cover
    tfd = tf.distributions


def main(args):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    # -----------------------------
    # Data: metadata + filtering + split
    # -----------------------------
    data_dir = Path(args.data_dir).expanduser().resolve()
    meta_df = ham10000.load_metadata(data_dir=data_dir, metadata_csv=args.metadata_csv)

    meta_df = ham10000.apply_filters(
        meta_df,
        dx=args.dx,
        localization=args.localization,
        dx_type=args.dx_type,
        sex=args.sex,
        age_min=args.age_min,
        age_max=args.age_max,
    )

    if args.max_samples is not None:
        # stable subset for fast debugging
        rng = np.random.RandomState(args.seed or 0)
        meta_df = meta_df.sample(n=min(args.max_samples, len(meta_df)), random_state=rng).reset_index(drop=True)

    if len(meta_df) == 0:
        raise ValueError("After filtering, dataset is empty. Check --dx/--localization filters.")

    # Train/test split
    if args.no_group_split:
        # random per-image split
        rng = np.random.RandomState(args.split_seed)
        perm = rng.permutation(len(meta_df))
        n_test = max(1, int(round(len(meta_df) * args.test_size)))
        test_idx = set(perm[:n_test].tolist())
        test_df = meta_df.iloc[list(test_idx)].copy().reset_index(drop=True)
        train_df = meta_df.drop(index=list(test_idx)).copy().reset_index(drop=True)
    else:
        train_df, test_df = ham10000.split_train_test_by_lesion(
            meta_df, test_size=args.test_size, seed=args.split_seed
        )

    # Labels (soft vectors)
    y_train = ham10000.make_soft_labels(train_df, num_labels=args.num_labels,
                                        off_value=args.label_off, on_value=args.label_on)
    y_test = ham10000.make_soft_labels(test_df, num_labels=args.num_labels,
                                       off_value=args.label_off, on_value=args.label_on)
    y_all = ham10000.make_soft_labels(meta_df, num_labels=args.num_labels,
                                      off_value=args.label_off, on_value=args.label_on)

    # We include image_id in the dataset so we can recover which sample was closest to an archetype.
    train_paths = train_df["filepath"].astype(str).tolist()
    test_paths = test_df["filepath"].astype(str).tolist()
    all_paths = meta_df["filepath"].astype(str).tolist()

    train_ids = train_df["image_id"].astype(str).tolist()
    test_ids = test_df["image_id"].astype(str).tolist()
    all_ids = meta_df["image_id"].astype(str).tolist()

    # For quick lookups in interpolation etc.
    id_to_path = {row["image_id"]: row["filepath"] for _, row in meta_df.iterrows()}

    print(f"[HAM10000] Loaded {len(meta_df)} samples after filtering.")
    print(f"[HAM10000] Train: {len(train_df)} | Test: {len(test_df)}")
    if args.dx:
        print(f"[HAM10000] dx filter: {args.dx}")
    if args.localization:
        print(f"[HAM10000] localization filter: {args.localization}")

    img_size = int(args.img_size)
    data_shape = [img_size, img_size, 3]

    # -----------------------------
    # tf.data input pipeline (streaming)
    # -----------------------------
    def _read_file(path):
        # TF1 compat
        return tf.read_file(path)

    def _decode_and_resize(path, label, image_id):
        img_bytes = _read_file(path)

        # Support jpg/png
        is_png = tf.strings.regex_full_match(tf.strings.lower(path), ".*\\.png")
        img = tf.cond(
            is_png,
            lambda: tf.image.decode_png(img_bytes, channels=3),
            lambda: tf.image.decode_jpeg(img_bytes, channels=3),
        )

        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize_images(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
        img.set_shape([img_size, img_size, 3])

        if args.augment and not args.test_model:
            img = tf.image.random_flip_left_right(img)

        return img, label, image_id

    def make_dataset(paths, labels, ids, training):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels, ids))
        if training:
            buf = min(len(paths), args.shuffle_buffer)
            ds = ds.shuffle(buffer_size=max(1, buf), seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.map(_decode_and_resize, num_parallel_calls=args.num_parallel_calls)
        ds = ds.batch(args.batch_size, drop_remainder=False)
        ds = ds.repeat()
        ds = ds.prefetch(args.prefetch)
        return ds

    train_ds = make_dataset(train_paths, y_train, train_ids, training=True)
    test_ds = make_dataset(test_paths, y_test, test_ids, training=False)
    all_ds = make_dataset(all_paths, y_all, all_ids, training=False)

    train_it = train_ds.make_one_shot_iterator()
    test_it = test_ds.make_one_shot_iterator()
    all_it = all_ds.make_one_shot_iterator()

    handle = tf.placeholder(tf.string, shape=[], name="dataset_handle")
    iterator = tf.data.Iterator.from_string_handle(handle, train_ds.output_types, train_ds.output_shapes)
    batch_x, batch_y, batch_id = iterator.get_next()

    train_handle = sess.run(train_it.string_handle())
    test_handle = sess.run(test_it.string_handle())
    all_handle = sess.run(all_it.string_handle())

    # Placeholders for "manual" inference (e.g., interpolation by image_id)
    data_ph = tf.placeholder(tf.float32, [None] + data_shape, name="data_ph")
    latent_code_ph = tf.placeholder(tf.float32, [None, args.dim_latentspace], name="latent_code")

    kl_loss_factor = tf.Variable(args.kl_loss_factor, dtype=tf.float32, trainable=False, name="kl_loss_factor")

    # -----------------------------
    # Model: prior, encoder, decoder
    # -----------------------------
    prior = lib_vae.build_prior(args.dim_latentspace)
    z_fixed_np = lib_at.create_z_fix(args.dim_latentspace)
    z_fixed = tf.constant(z_fixed_np, dtype=tf.float32)

    if args.vae:
        encoder = lib_vae.build_encoder_vae(args.dim_latentspace, x_shape=data_shape)
        encoded_batch = encoder(batch_x)
        encoded_ph = encoder(data_ph)
        t_posterior = encoded_batch["p"]
        mu_t = encoded_batch["mu"]
        mu_ph = encoded_ph["mu"]
        z_predicted = None
    else:
        if args.encoder_arch == "basic":
            encoder = lib_vae.build_encoder_basic(args.dim_latentspace, z_fixed)
        else:
            encoder = lib_vae.build_encoder_convs(args.dim_latentspace, z_fixed, x_shape=data_shape)
        encoded_batch = encoder(batch_x)
        encoded_ph = encoder(data_ph)
        t_posterior = encoded_batch["p"]
        mu_t = encoded_batch["mu"]
        mu_ph = encoded_ph["mu"]
        z_predicted = encoded_batch["z_predicted"]

    decoder = lib_vae.build_decoder(
        data_shape=data_shape,
        num_labels=args.num_labels,
        trainable_var=args.trainable_var,
        init_stddev=args.decoder_init_stddev,
    )

    decoded_from_data = decoder(mu_t)
    x_hat = decoded_from_data["x_hat"]
    y_hat = decoded_from_data["side_info"][:, :args.num_labels]

    decoded_from_latent = decoder(latent_code_ph)
    latent_decoded_x = decoded_from_latent["x_hat"]
    latent_decoded_y = decoded_from_latent["side_info"][:, :args.num_labels]

    # -----------------------------
    # Loss (kept as in your JAFFE script)
    # -----------------------------
    def build_loss():
        likelihood = tf.reduce_mean(x_hat.log_prob(batch_x))

        if args.dir_prior:
            q_sample = t_posterior.sample(50)
            # Monte Carlo estimate of KL(q||p)
            divergence = tf.reduce_mean(t_posterior.log_prob(q_sample) - prior.log_prob(q_sample))
        else:
            divergence = tf.reduce_mean(tfd.kl_divergence(t_posterior, prior))

        if not args.vae:
            archetype_loss = tf.losses.mean_squared_error(z_predicted, z_fixed)
        else:
            archetype_loss = tf.constant(0.0, dtype=tf.float32)

        class_loss = tf.losses.mean_squared_error(
            predictions=y_hat,
            labels=batch_y
        )

        elbo = tf.reduce_mean(
            args.recon_loss_factor * likelihood
            - args.class_loss_factor * class_loss
            - args.at_loss_factor * archetype_loss
            - kl_loss_factor * divergence
        )

        return archetype_loss, class_loss, likelihood, divergence, elbo

    archetype_loss, class_loss, likelihood, divergence, elbo = build_loss()

    lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(-elbo)

    # Priors for sampling
    dirichlet_prior = lib_vae.dirichlet_prior(dim_latentspace=args.dim_latentspace, alpha=args.dirichlet_alpha)
    num_prior_samples = tf.placeholder(tf.int32, (), name='num_prior_samples')

    if args.vae:
        samples_prior = prior.sample(num_prior_samples, seed=113)
        samples_decoded = decoder(samples_prior)
    else:
        samples_prior = dirichlet_prior.sample(num_prior_samples, seed=113)
        samples_decoded = decoder(tf.matmul(samples_prior, z_fixed))

    samples_decoded = {
        "x_hat": tf.stop_gradient(samples_decoded["x_hat"].mean()),
        "side_info": tf.stop_gradient(samples_decoded["side_info"])
    }

    # Summaries
    tf.summary.scalar('elbo', elbo)
    tf.summary.scalar('archetype_loss', archetype_loss)
    tf.summary.scalar('sideinfo_loss', class_loss)
    tf.summary.scalar('likelihood', likelihood)
    tf.summary.scalar('kl_divergence', divergence)
    tf.summary.scalar('learning_rate', lr)

    hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in vars(args).items()]
    tf.summary.text('hyperparameters', tf.stack(hyperparameters))

    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    # -----------------------------
    # Plot / save helpers (adapted for on-disk data)
    # -----------------------------
    def plot_latent_traversal(filename=None, title="Latent Traversal", traversal_steps_per_dir=15, z_dims=None):
        traversal_weights, _ = lib_at.barycentric_coords(n_per_axis=traversal_steps_per_dir)
        z_f = z_fixed_np.copy()

        if z_f.shape[0] > 3:
            if z_dims is None:
                z_dims = [0, 1, 2]
            z_f = z_f[z_dims, :]
        elif z_f.shape[0] < 3:
            traversal_weights = traversal_weights[:traversal_steps_per_dir, 1:]

        traversal_latents = np.dot(traversal_weights, z_f)
        imgs_traversal = sess.run(latent_decoded_x.mean(), feed_dict={latent_code_ph: traversal_latents})

        fig = lib_plt.grid_plot(
            imgs_sampled=imgs_traversal,
            n_perAxis=traversal_steps_per_dir,
            px=data_shape[0],
            py=data_shape[1],
            channels=3,
            figSize=16,
            title=title
        )

        if filename is None:
            filename = 'latent_traversal_final.png'
        fig.savefig(FINAL_RESULTS_DIR / filename, dpi=300)
        plt.close(fig)

    def _collect_latents(dataset_handle, n_samples):
        """Collect latent means mu for n_samples from the chosen dataset handle."""
        num_steps = int(math.ceil(n_samples / args.batch_size))
        latents = []
        ids = []
        labels = []

        for _ in range(num_steps):
            mu_b, id_b, y_b = sess.run(
                [mu_t, batch_id, batch_y],
                feed_dict={handle: dataset_handle}
            )
            latents.append(mu_b)
            ids.append(id_b)
            labels.append(y_b)

        latents = np.vstack(latents)[:n_samples]
        ids = np.concatenate(ids)[:n_samples]
        labels = np.vstack(labels)[:n_samples]
        return latents, ids, labels

    def plot_z_fixed(path, plot_generated=False):
        """
        Plot archetypes (either generated or closest real images) + latent scatter.
        """
        latents_test, ids_test, y_test_vec = _collect_latents(test_handle, n_samples=len(test_df))

        if plot_generated:
            image_z_fixed, label_z_fixed = sess.run(
                [latent_decoded_x.mean(), latent_decoded_y],
                feed_dict={latent_code_ph: z_fixed_np}
            )
            label_z_fixed = np.argmax(label_z_fixed, axis=1)
        else:
            idx_closest_to_at = []
            for i in range(z_fixed_np.shape[0]):
                distances = np.linalg.norm(latents_test - z_fixed_np[i], axis=1)
                idx_closest_to_at.append(int(np.argmin(distances)))

            chosen_ids = [ids_test[i].decode("utf-8") if isinstance(ids_test[i], (bytes, bytearray)) else str(ids_test[i])
                          for i in idx_closest_to_at]

            image_z_fixed = np.stack(
                [ham10000.load_image_np(id_to_path[cid], img_size=img_size) for cid in chosen_ids],
                axis=0
            )
            label_z_fixed = np.argmax(y_test_vec[idx_closest_to_at, :], axis=1)

        scatterplot_labels = np.argmax(y_test_vec, axis=1)

        fig_zfixed = lib_plt.plot_samples(
            samples=image_z_fixed,
            latent_codes=latents_test,
            labels=scatterplot_labels,
            epoch=None,
            titles=[f"Archetype {i + 1}" for i in range(nAT)],
            img_labels=label_z_fixed
        )
        fig_zfixed.savefig(path, dpi=200)
        plt.close(fig_zfixed)

    def plot_random_samples(path, epoch=None, n_samples=49):
        tensors_rsample = [
            samples_decoded["x_hat"],
            samples_decoded["side_info"],
            samples_prior
        ]

        rnd_samples_img, rnd_samples_labels, rnd_samples_latents = sess.run(
            tensors_rsample,
            feed_dict={num_prior_samples: n_samples}
        )

        rnd_label_ids = np.argmax(rnd_samples_labels, axis=1)
        nrows = int(np.sqrt(n_samples))

        fig_rsamples = lib_plt.plot_samples(
            samples=rnd_samples_img,
            latent_codes=rnd_samples_latents,
            labels=rnd_label_ids,
            nrows=nrows,
            epoch=epoch
        )

        fig_rsamples.savefig(path, dpi=300)
        plt.close(fig_rsamples)

    def plot_hinton(weight_target=0.65):
        if z_fixed_np.shape[0] != 3:
            print("[plot_hinton] Only supported for 3 archetypes (dim_latentspace=2). Skipping.")
            return

        other_weights = (1.0 - weight_target) / 2.0
        weights = np.array([
            [1.0 / 3, 1.0 / 3, 1.0 / 3],
            [other_weights, other_weights, weight_target],
            [other_weights, weight_target, other_weights],
            [weight_target, other_weights, other_weights]
        ], dtype=np.float32)

        latent_coords = weights @ z_fixed_np

        imgs_hinton = sess.run(
            latent_decoded_x.mean(),
            feed_dict={latent_code_ph: latent_coords}
        )

        samples_z_f = sess.run(
            latent_decoded_x.mean(),
            feed_dict={latent_code_ph: z_fixed_np}
        )

        fig = lib_at.create_hinton_plot(
            z_sampled=samples_z_f,
            weightMatrix=weights,
            mixture_sampled=imgs_hinton,
            figSize=8
        )

        fig.savefig(
            FINAL_RESULTS_DIR / f"hinton_{weight_target:.2f}.png",
            dpi=200
        )
        plt.close(fig)

    def plot_interpolation(start_image_id, end_image_id, nb_samples=9, nb_rows=3, nb_cols=3):
        assert nb_rows * nb_cols >= nb_samples

        if start_image_id not in id_to_path:
            raise ValueError(f"start_image_id={start_image_id} not found in current subset.")
        if end_image_id not in id_to_path:
            raise ValueError(f"end_image_id={end_image_id} not found in current subset.")

        img_1 = ham10000.load_image_np(id_to_path[start_image_id], img_size=img_size)[None, ...]
        img_2 = ham10000.load_image_np(id_to_path[end_image_id], img_size=img_size)[None, ...]

        z1 = sess.run(mu_ph, feed_dict={data_ph: img_1})
        z2 = sess.run(mu_ph, feed_dict={data_ph: img_2})

        latent_path = lib_plt.interpolate_points(coord_init=z1[0], coord_end=z2[0], nb_samples=nb_samples)

        imgs_interp, labels_interp = sess.run(
            [latent_decoded_x.mean(), latent_decoded_y],
            feed_dict={latent_code_ph: latent_path}
        )

        df_labels = pd.DataFrame(labels_interp, columns=ham10000.DX_CLASSES[:args.num_labels])
        df_labels.to_csv(
            FINAL_RESULTS_DIR / f"interpolation_{start_image_id}_to_{end_image_id}_labels.csv",
            index=False
        )

        fig = lib_at.plot_sample_path(
            samplePath_imgs=imgs_interp,
            nbRow=nb_rows,
            nbCol=nb_cols,
            figSize=10,
            data_shape=data_shape
        )

        fig.savefig(
            FINAL_RESULTS_DIR / f"interpolation_{start_image_id}_to_{end_image_id}.png",
            dpi=200
        )
        plt.close(fig)

    def create_latent_df():
        latents_all, ids_all, y_all_vec = _collect_latents(all_handle, n_samples=len(meta_df))
        cols_dims = [f"ldim{i}" for i in range(args.dim_latentspace)]
        df_latent = pd.DataFrame(latents_all, columns=cols_dims)

        # image_id as string
        ids_all_str = [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in ids_all]
        df_latent.insert(0, "image_id", ids_all_str)

        label_cols_all = ham10000.DX_CLASSES[:args.num_labels]
        df_labels = pd.DataFrame(y_all_vec, columns=label_cols_all)

        # Join with metadata (kept in the same order as all_ids)
        df_meta = meta_df.copy()
        # Keep only a small subset of metadata columns to avoid very wide files by default
        keep_cols = [c for c in ["lesion_id", "dx", "dx_type", "age", "sex", "localization"] if c in df_meta.columns]
        df_meta = df_meta[["image_id"] + keep_cols]

        df = df_latent.merge(df_meta, on="image_id", how="left")
        df = pd.concat([df, df_labels], axis=1)
        return df

    # -----------------------------
    # Results directories (fixed to use args.results_path)
    # -----------------------------
    CUR_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = Path(args.results_path).expanduser().resolve()

    if not args.test_model:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        MODEL_DIR = RESULTS_DIR / f"{timestamp}_{args.runNB}_{args.dim_latentspace}_{args.batch_size}_{args.n_epochs}"
    else:
        all_results = os.listdir(str(RESULTS_DIR))
        if args.model_substr is not None:
            idx = [args.model_substr in res for res in all_results]
            all_results = list(compress(all_results, idx))
        all_results.sort()
        if not all_results:
            raise FileNotFoundError(f"No trained models found in {RESULTS_DIR}")
        MODEL_DIR = RESULTS_DIR / all_results[-1]

    global FINAL_RESULTS_DIR, TENSORBOARD_DIR, IMGS_DIR, SAVED_MODELS_DIR
    FINAL_RESULTS_DIR = MODEL_DIR / 'final_results/'
    TENSORBOARD_DIR = MODEL_DIR / 'Tensorboard'
    IMGS_DIR = MODEL_DIR / 'imgs'
    SAVED_MODELS_DIR = MODEL_DIR / 'Saved_models/'

    if not args.test_model:
        for path in [TENSORBOARD_DIR, SAVED_MODELS_DIR, IMGS_DIR]:
            os.makedirs(path, exist_ok=True)

        os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
        meta_df.to_csv(FINAL_RESULTS_DIR / "subset_metadata.csv", index=False)

    # -----------------------------
    # Training loop
    # -----------------------------
    num_mb_its_per_epoch = int(math.ceil(len(train_df) / args.batch_size))
    num_mb_its_per_epoch_test = int(math.ceil(len(test_df) / args.batch_size))

    saver = tf.train.Saver()
    step = 0

    adjusted_lr = False
    cur_kl_factor = 5000.0

    if not args.test_model:
        writer = tf.summary.FileWriter(str(TENSORBOARD_DIR), sess.graph)

        for epoch in range(args.n_epochs):

            # optional LR schedule (kept similar to your script)
            if epoch >= args.lr_drop_epoch and not adjusted_lr:
                sess.run(tf.assign(lr, args.lr_after_drop))
                adjusted_lr = True

            # -------- TRAIN --------
            for _ in range(num_mb_its_per_epoch):
                sess.run(
                    train_op,
                    feed_dict={
                        handle: train_handle,
                        kl_loss_factor: cur_kl_factor,
                    }
                )
                step += 1

            # -------- EVAL --------
            if epoch % args.test_frequency_epochs == 0:
                cur_kl_factor = max(cur_kl_factor / args.kl_decrease_factor, args.kl_loss_factor)

                tensors_test = [summary_op, elbo, likelihood, divergence, archetype_loss, class_loss]

                test_elbo = test_like = test_div = test_at = test_cls = 0.0

                for _ in range(num_mb_its_per_epoch_test):
                    summary, e, l, d, a, c = sess.run(
                        tensors_test,
                        feed_dict={
                            handle: test_handle,
                            kl_loss_factor: cur_kl_factor,
                        }
                    )
                    writer.add_summary(summary, step)

                    test_elbo += e
                    test_like += l
                    test_div += d
                    test_at += a
                    test_cls += c

                test_elbo /= num_mb_its_per_epoch_test
                test_like /= num_mb_its_per_epoch_test
                test_div /= num_mb_its_per_epoch_test
                test_at /= num_mb_its_per_epoch_test
                test_cls /= num_mb_its_per_epoch_test

                print(
                    f"\nEpoch {epoch}\n"
                    f"ELBO: {test_elbo}\n"
                    f"Likelihood: {test_like}\n"
                    f"Divergence: {test_div}\n"
                    f"Archetype loss: {test_at}\n"
                    f"Class loss: {test_cls}\n"
                )

                # Save a couple diagnostic plots
                os.makedirs(IMGS_DIR, exist_ok=True)
                plot_z_fixed(IMGS_DIR / f"Z_fixed_epoch{epoch}.png", plot_generated=True)

            # -------- SAVE --------
            if epoch % args.save_each == 0 and epoch > 0:
                saver.save(sess, save_path=str(SAVED_MODELS_DIR / "save"), global_step=epoch)

        print("Training finished.")

        os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

        # final plots & latents
        plot_z_fixed(FINAL_RESULTS_DIR / "Z_fixed_final_closest.png", plot_generated=False)
        plot_z_fixed(FINAL_RESULTS_DIR / "Z_fixed_final_generated.png", plot_generated=True)

        df_lat = create_latent_df()
        df_lat.to_csv(FINAL_RESULTS_DIR / "latent_codes.csv", index=False)

        plot_latent_traversal()

        if args.make_hinton:
            plot_hinton(weight_target=args.hinton_weight)

    else:
        # -----------------------------
        # Inference mode
        # -----------------------------
        saver.restore(sess, tf.train.latest_checkpoint(str(SAVED_MODELS_DIR)))
        os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

        df_lat = create_latent_df()
        df_lat.to_csv(FINAL_RESULTS_DIR / "latent_codes.csv", index=False)

        plot_latent_traversal()
        plot_z_fixed(FINAL_RESULTS_DIR / "Z_fixed_final_closest.png", plot_generated=False)
        plot_z_fixed(FINAL_RESULTS_DIR / "Z_fixed_final_generated.png", plot_generated=True)
        if args.make_hinton:
            plot_hinton(weight_target=args.hinton_weight)

        if args.interp_from is not None and args.interp_to is not None:
            plot_interpolation(args.interp_from, args.interp_to, nb_samples=args.interp_steps)

    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # -----------------------------
    # Dataset / filtering
    # -----------------------------
    parser.add_argument('--data-dir', type=str, required=True,
                        help="Path to the HAM10000 root folder containing the images (part_1/part_2) and metadata CSV.")
    parser.add_argument('--metadata-csv', type=str, default=None,
                        help="Optional: explicit path to HAM10000_metadata.csv. If omitted, auto-detected in --data-dir.")
    parser.add_argument('--img-size', type=int, default=64, help="Resize images to img-size x img-size. Use 64 or 128.")
    parser.add_argument('--augment', action='store_true', default=False, help="Apply simple augmentation (random horizontal flip).")

    parser.add_argument('--dx', nargs='*', default=None,
                        help="Filter by dx label(s). Example: --dx mel  OR  --dx mel nv")
    parser.add_argument('--localization', nargs='*', default=None,
                        help="Filter by localization(s). Example: --localization scalp back")
    parser.add_argument('--dx-type', nargs='*', default=None,
                        help="Filter by dx_type(s). Example: --dx-type histo")
    parser.add_argument('--sex', nargs='*', default=None,
                        help="Filter by sex. Example: --sex male female")
    parser.add_argument('--age-min', type=float, default=None)
    parser.add_argument('--age-max', type=float, default=None)
    parser.add_argument('--max-samples', type=int, default=None,
                        help="Debug only: limit number of samples after filtering.")

    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--no-group-split', action='store_true', default=False,
                        help="If set, split per-image (may leak lesion_id across train/test). Default: group split by lesion_id.")

    # Label smoothing (kept like your JAFFE loader)
    parser.add_argument('--label-off', type=float, default=0.03)
    parser.add_argument('--label-on', type=float, default=0.82)

    # -----------------------------
    # Logging / results
    # -----------------------------
    parser.add_argument('--runNB', type=str, default="1")
    parser.add_argument('--results-path', type=str, default='./Results/HAM10000')
    parser.add_argument('--test-frequency-epochs', type=int, default=50)
    parser.add_argument('--save_each', type=int, default=500)

    # -----------------------------
    # NN settings
    # -----------------------------
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=1001)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--num-archetypes', type=int, default=None,
                        help="Optional convenience: set k directly. Then latent dim = k-1. Overrides --dim-latentspace.")
    parser.add_argument('--dim-latentspace', type=int, default=2,
                        help="Number of Archetypes = Latent Space Dimension + 1")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_labels', type=int, default=7)

    parser.add_argument('--encoder-arch', type=str, default='convs', choices=['convs', 'basic'],
                        help="Encoder architecture. 'convs' is closer to the paper; 'basic' matches your shared script.")

    parser.add_argument('--trainable-var', action='store_true', default=False,
                        help="If set, learn decoder observation stddev (Normal scale).")
    parser.add_argument('--decoder-init-stddev', type=float, default=0.1)

    # Data pipeline knobs
    parser.add_argument('--num-parallel-calls', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--shuffle-buffer', type=int, default=2000)

    # DAA loss: weights
    parser.add_argument('--at-loss-factor', type=float, default=100.0)
    parser.add_argument('--class-loss-factor', type=float, default=200.0)
    parser.add_argument('--recon-loss-factor', type=float, default=0.4)
    parser.add_argument('--kl-loss-factor', type=float, default=40.0)
    parser.add_argument('--kl-decrease-factor', type=float, default=1.5)

    # LR schedule
    parser.add_argument('--lr-drop-epoch', type=int, default=2000)
    parser.add_argument('--lr-after-drop', type=float, default=5e-4)

    # loading already existing model
    parser.add_argument('--test-model', dest='test_model', action='store_true', default=False)
    parser.add_argument('--model-substr', type=str, default=None)

    # Different settings for the prior
    parser.add_argument('--dir-prior', dest='dir_prior', action='store_true', default=False,
                        help="Use Monte-Carlo estimate of KL(q||p).")
    parser.add_argument('--dirichlet-alpha', type=float, default=0.7,
                        help="Alpha for Dirichlet sampling when generating random samples (not the KL prior).")
    parser.add_argument('--vae', dest='vae', action='store_true', default=False,
                        help="Train standard VAE instead of DAA (no archetype loss).")

    # Optional: diagnostics
    parser.add_argument('--make-hinton', action='store_true', default=False)
    parser.add_argument('--hinton-weight', type=float, default=0.65)

    parser.add_argument('--interp-from', type=str, default=None, help="Optional: image_id to start interpolation.")
    parser.add_argument('--interp-to', type=str, default=None, help="Optional: image_id to end interpolation.")
    parser.add_argument('--interp-steps', type=int, default=9)

    args = parser.parse_args()

    # GPU target
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # model hyperparams
    if args.num_archetypes is not None:
        if args.num_archetypes < 2:
            raise ValueError("--num-archetypes must be >= 2")
        args.dim_latentspace = int(args.num_archetypes) - 1

    if not (1 <= args.num_labels <= 7):
        raise ValueError("HAM has 7 labels: set --num_labels <= 7")

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    global nAT
    nAT = args.dim_latentspace + 1

    print(args)
    main(args)
