import contextlib
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:  # pragma: no cover
    sns = None
    _HAS_SEABORN = False

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _axes_style_ctx():
    if _HAS_SEABORN:
        return sns.axes_style("darkgrid")
    return contextlib.nullcontext()


def interpolate_points(coord_init, coord_end, nb_samples):
    """
    Utility function for interpolation between two points.
    :param coord_init:
    :param coord_end:
    :param nb_samples:
    :return:
    """
    interpol = np.zeros([nb_samples, coord_init.shape[-1]])
    for alpha in range(nb_samples):
        interpol[alpha, :] = (alpha / (nb_samples - 1)) * (coord_end - coord_init) + coord_init
    return interpol


def plot_samples(samples, latent_codes, labels,
                 epoch, nrows=1,
                 titles=None, size=2, latent_ticks=False,
                 img_labels=None):
    """
    Plot:
      - If latent_codes is 2D: 2D scatter plot + sampled images
      - If latent_codes is 3D: 3D scatter plot + sampled images
      - Else: only sampled images
    """
    assert titles is None or len(titles) == len(samples)

    xticks = np.arange(-1, 1.1, 0.25)
    yticks = np.arange(-0.75, 1.30, 0.25)

    ncols = int(np.ceil(len(samples) / nrows))
    if latent_codes is not None and latent_codes.ndim == 2 and latent_codes.shape[1] <= 3 and ncols * nrows < len(samples) + 1:
        ncols += 1

    fig = plt.figure(figsize=(ncols * size, nrows * size))
    no_ticks = dict(left=False, bottom=False, labelleft=False, labelbottom=False)
    title = ''

    colors = [
        "#00CB50",  # melanoma
        "#F350C0",  # nevus
        "#00D8FC",  # benign keratosis
        "#EC001C",  # basal cell carcinoma
        "#9230DD",  # actinic keratosis
        "#FF9800",  # vascular lesion
        "#4CAF50"   # dermatofibroma
    ]

    label_str = [
        "melanoma",
        "nevus",
        "benign keratosis",
        "basal cell carcinoma",
        "actinic keratosis",
        "vascular lesion",
        "dermatofibroma"
    ]

    markers = ["o", "^", "s", "D", "p", "X", "*"]

    offset = 0
    if latent_codes is not None and latent_codes.ndim == 2 and latent_codes.shape[1] <= 3:
        with _axes_style_ctx():
            if latent_codes.shape[1] == 3:
                ax = fig.add_subplot(nrows, ncols, 1, projection='3d')
                if epoch is not None:
                    title = f'Epoch {epoch}'
                # 3D scatter
                for lab in np.unique(labels):
                    coords = latent_codes[labels == lab, ...]
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               s=3, c=np.repeat(colors[int(lab)], coords.shape[0]),
                               label=label_str[int(lab)], alpha=0.5,
                               marker=markers[int(lab)])
                ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(-0.7, 0.5), markerscale=4)
                ax.set_title(title, fontsize=8)

            else:
                ax = fig.add_subplot(nrows, ncols, 1)
                if epoch is not None:
                    ax.set_ylabel(f'Epoch {epoch}')
                ax.set_aspect('equal')

                # classic 2D simplex for 3 archetypes
                Z_fix = np.array([[1., 0.],
                                  [-0.5, 0.8660254],
                                  [-0.5, -0.8660254],
                                  [1., 0.]])

                for lab in np.unique(labels):
                    coords = latent_codes[labels == lab, ...]
                    ax.scatter(coords[:, 1], coords[:, 0],
                               s=3, c=np.repeat(colors[int(lab)], coords.shape[0]),
                               label=label_str[int(lab)], alpha=0.5,
                               marker=markers[int(lab)])
                ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(-0.7, 0.5), markerscale=4)

                ax.scatter(Z_fix[:-1, 1], Z_fix[:-1, 0], color='darkred', marker='x', s=8)
                ax.plot(Z_fix[:, 1], Z_fix[:, 0], color='darkblue', linestyle=(0, (2, 1, 2, 1)), linewidth=1)

                ax.set_xlim(np.min(xticks), np.max(xticks))
                ax.set_ylim(np.min(yticks), np.max(yticks))
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
                if not latent_ticks:
                    ax.tick_params(axis='both', which='both', **no_ticks)
                else:
                    ax.tick_params(axis='both', which='both', labelsize=4)

                ax.set_ylim(ax.get_ylim()[::-1])
        offset = 1

    title_idx = 0
    for index, sample in enumerate(samples):
        ax = fig.add_subplot(nrows, ncols, offset + index + 1)

        if sample.ndim == 2:
            ax.imshow(sample, cmap='gray')
        else:
            ax.imshow(sample)

        if titles is not None:
            ax.set_title(titles[title_idx], ha='center', va='center', alpha=.8, size=7)
            title_idx += 1

        if img_labels is not None:
            for spine in ax.spines.values():
                spine.set_edgecolor(colors[int(img_labels[index])])
                spine.set_linewidth(4)

        ax.tick_params(axis='both', which='both', **no_ticks)

    fig.suptitle(title, fontsize=8)
    return fig


def plot_video_img(image_point, latent_point, label_point, z_fixed_, labels_z_fixed, df_labels):
    """
    Used for GIF creation: shows simplex + current point + image + label bars.
    Falls back to matplotlib-only if seaborn isn't installed.
    """
    if _HAS_SEABORN:
        palette = sns.color_palette("tab10", 7)
    else:
        palette = plt.get_cmap("tab10").colors

    def get_color(label):
        if isinstance(label, (int, np.integer)):
            return palette[int(label)]
        else:
            return palette[int(np.argmax(label))]

    f = plt.figure(figsize=(15, 5))
    ax = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)

    simplex = plt.Polygon(z_fixed_, facecolor="whitesmoke",
                          edgecolor="black", lw=1, zorder=0)
    ax.add_patch(simplex)
    ax.set_xlim((-0.75, 1.25))
    ax.set_ylim((-1, 1))
    ax.axis('off')
    ax.set_aspect('equal')

    # archetypes
    for i in range(3):
        ax.scatter(z_fixed_[i, 0], z_fixed_[i, 1],
                   color=get_color(labels_z_fixed[i]),
                   marker='*', s=80)
        ax.annotate(f"AT{i+1}", z_fixed_[i] + np.array([0.05, 0]))

    # current latent point
    ax.scatter(latent_point[0], latent_point[1],
               color=get_color(label_point), s=150)

    for spine in ax2.spines.values():
        spine.set_edgecolor(get_color(label_point))
        spine.set_linewidth(4)
    ax2.axis('off')

    if image_point.ndim == 2:
        ax2.imshow(image_point, cmap="gray")
    else:
        ax2.imshow(image_point)

    # label bars
    if _HAS_SEABORN:
        sns.barplot(data=df_labels, x='class', y='score', ax=ax3)
    else:
        ax3.bar(df_labels["class"], df_labels["score"])
        ax3.set_xticklabels(df_labels["class"], rotation=90, fontsize=8)

    ax3.set_ylim((0, 1))
    ax3.set_ylabel("Probability")

    f.canvas.draw()
    image = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
    plt.close(f)

    return image


def grid_plot(imgs_sampled, n_perAxis, px=128, py=128, figSize=16, channels=3, title=""):
    if channels == 1:
        imgMatrix = np.ones([n_perAxis * px, n_perAxis * py])
        cmap = "gray"
    else:
        imgMatrix = np.ones([n_perAxis * px, n_perAxis * py, channels])
        cmap = None

    nbCols = np.zeros(n_perAxis, dtype=int)
    cnt = 0
    imgC = 0
    for i in range(n_perAxis, 0, -1):
        nbCols[cnt] = i
        cnt += 1

    for i in range(n_perAxis):
        ccc = 0
        for j in range(nbCols[i]):
            x_start = int(0.5 * j * px + i * px)
            x_end = int(0.5 * j * px + (i + 1) * px)
            y_start = ccc * py
            y_end = (ccc + 1) * py

            if channels == 1:
                imgMatrix[x_start:x_end, y_start:y_end] = imgs_sampled[imgC]
            else:
                imgMatrix[x_start:x_end, y_start:y_end, :] = imgs_sampled[imgC]

            ccc += 1
            imgC += 1

    fig = plt.figure(figsize=(figSize, figSize))
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    ax.tick_params(axis='both', which='both',
                   left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=1.6 * figSize)
    ax.imshow(imgMatrix, cmap=cmap)

    return fig
