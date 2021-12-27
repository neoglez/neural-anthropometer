import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image


t = transforms.ToPILImage()
# the width is 20 cm, which is almost as wide as a a4.
figure_two_colums_width = 7.87
# the height keeps a pleasant 1.6 ratio
figure_two_colums_height = 4.0
figure_title_fontsize = 10
tables_fontsize = 10
axes_title_fontsize = 10


def annotations_to_latex_table(
    actuals, predicted=None, cm=True, percentage=True
):
    if predicted is None:
        predicted = np.array([0 for i in range(8)])

    if torch.is_tensor(actuals) and not torch.is_tensor(predicted):
        predicted = torch.tensor(predicted)

    if cm:
        actuals = actuals * 100
        predicted = predicted * 100

    error = predicted - actuals
    if percentage:
        error = (error / actuals) * 100

    table_head = (
        r"\begin{tabular}{ | c | c | c | c |} "
        + r"  \hline "
        + r"  \textbf{HBD} & \textbf{Actual} & \textbf{Predicted} & "
    )
    table_head_error = (
        r"  \textbf{Error} \\ "
    )
    if percentage:
        table_head_error = (
            r"  \textbf{Error (\%)} \\ "
        )
    table_body = (
        r"  \hline	CC & {:.2f} & {:.2f} & {:+.2f} \\ "
        + r"  \hline H & {:.2f} & {:.2f} & {:+.2f} \\ "
        + r"  \hline I & {:.2f} & {:.2f} & {:+.2f} \\ "
        + r"  \hline LAL & {:.2f} & {:.2f} & {:+.2f} \\ "
        + r"  \hline PC & {:.2f} & {:.2f} & {:+.2f} \\ "
        + r"  \hline RAL & {:.2f} & {:.2f} & {:+.2f} \\ "
        + r"  \hline SW & {:.2f} & {:.2f} & {:+.2f} \\ "
        + r"  \hline WC & {:.2f} & {:.2f} & {:+.2f} \\"
    ).format(
        actuals[0],
        predicted[0],
        error[0],
        actuals[1],
        predicted[1],
        error[1],
        actuals[2],
        predicted[2],
        error[2],
        actuals[3],
        predicted[3],
        error[3],
        actuals[4],
        predicted[4],
        error[4],
        actuals[5],
        predicted[5],
        error[5],
        actuals[6],
        predicted[6],
        error[6],
        actuals[7],
        predicted[7],
        error[0],
    )

    table_footer = r" \hline\end{tabular}"
    return table_head + table_head_error + table_body + table_footer


def to_pil_image(nimage):
    """
    Convert to PIL Image using torchvision transform

    Parameters
    ----------
    nimage : tensor or numpy ndarray of shape 1 x H x W
        image tensor

    Returnst
    -------
    PIL Image H x W

    """
    return t(nimage)


def image_grid(
    subject_images,
    actuals=None,
    predicted=None,
    subject_matada=None,
    background="gray",
):
    """
    Make a grid of images and annotations

    Parameters
    ----------
    subject_images : Tensor or list
        4D mini-batch Tensor of
        shape (B x C x H x W) or a list of images all of the same size.
    annotations : Tensor or list, optional
        The default is None.
    background : string, optional
        One of "gray", "black", "yellow" or
        "white". The default is "gray".

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    if not (
        torch.is_tensor(subject_images)
        or (
            isinstance(subject_images, list)
            and all(torch.is_tensor(t) for t in subject_images)
        )
    ):
        raise TypeError(
            "tensor or list of tensors expected, got {}".format(
                type(subject_images)
            )
        )

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(subject_images, list):
        subject_images = torch.stack(subject_images, dim=0)

    l = subject_images.shape[0]

    if predicted is None:
        predicted = np.zeros(actuals.shape)

    matplotlib.rc("text", usetex=True)  # use latex for text
    # add amsmath to the preamble
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    fig, ax = plt.subplots(
        2, l, figsize=(figure_two_colums_width, figure_two_colums_height)
    )
    # fig.tight_layout(pad=1)
    #left  = 0.125  # the left side of the subplots of the figure
    #right = 0.9    # the right side of the subplots of the figure
    #bottom = 0.1   # the bottom of the subplots of the figure
    #top = 0.9      # the top of the subplots of the figure
    #wspace = 0.2   # the amount of width reserved for blank space between subplots
    #hspace = 0.2   # the amount of height reserved for white space between subplots


    fig.subplots_adjust(
        left=0.05, bottom=0.05, right=None, top=None, wspace=0.5, hspace=0.1
    )

    fig.suptitle(
        r"\textbf{HBD inference results on 4 subjects}", fontsize="x-large"
    )
    # Plot the images in top row
    for i in range(l):
        if background == "gray":
            img = to_pil_image(subject_images[i]).convert("RGB")
            ax[0][i].imshow(img)
        elif background == "black":
            img = subject_images[i].squeeze()
            ax[0][i].imshow(img, cmap="Greys")
        elif background == "yellow":
            img = subject_images[i].squeeze()
            ax[0][i].imshow(img)
        elif background == "white":
            img = subject_images[i].squeeze()
            ax[0][i].imshow(img, cmap="gray")

        ax[0][i].set_title(subject_matada[i], fontsize=axes_title_fontsize)
        ax[1][i].axis("off")
        ax[1][i].text(
            0,
            0.7,
            annotations_to_latex_table(actuals[i], predicted=predicted[i]),
            fontsize=tables_fontsize,
        )

    return fig


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":

    import neural_anthropometer as na
    from torch.utils.data.dataset import random_split
    from torch.utils.data import DataLoader
    import os
    from torchvision import transforms

    transform = na.TwoDToTensor()
    to_pil_image = transforms.ToPILImage(mode="L")

    rootDir = os.path.abspath(os.path.curdir)
    datasetDir = os.path.join(rootDir, "..", "..", "dataset")
    format_path = os.path.join(rootDir, "..", "..", "format")

    plt.style.use(
        os.path.join(format_path, "PaperDoubleFig.mplstyle")
    )

    batch_size = 4

    # 2700 instances to train and validate
    na_train = na.NeuralAnthropometerSyntheticImagesDatasetTrainTest(
        datasetDir, train=True, transform=transform
    )

    # Get a mini-batch of 4 images to display
    train_dt = DataLoader(na_train, batch_size=batch_size, shuffle=True)
    dataiter = iter(train_dt)
    # Get some random training images and annotations.
    # The tensor has following structure:
    #
    # for _, data in enumerate(train_dt):
    #   actual_hbds = data["annotations"]["human_dimensions"]
    #   inputs = data["image"]
    # Furthermore, the structure of actual_hbds dict is:
    # {'chest_circumference': tensor([1.0164], dtype=torch.float64),
    #  'height': tensor([1.8133], dtype=torch.float64),
    #  'inseam': tensor([0.8059], dtype=torch.float64),
    #  'left_arm_length': tensor([0.5784], dtype=torch.float64),
    #  'pelvis_circumference': tensor([1.0575], dtype=torch.float64),
    #  'right_arm_length': tensor([0.6005], dtype=torch.float64),
    #  'shoulder_width': tensor([0.4087], dtype=torch.float64),
    #  'waist_circumference': tensor([0.8459], dtype=torch.float64)
    #  }
    # The equivalent tensor contains the information in the corresponding
    # integer indices. It is important to remember that all HBD are given in
    # meters. If you want to convert them to cm, you have to multiply by 100.
    data = dataiter.next()
    actual_hbds = data["annotations"]["human_dimensions"]
    images = data["image"]
    metadata = data["subject_string"]
    # create grid of images and annotations
    fig = image_grid(
        images, actual_hbds, subject_matada=metadata, background="white"
    )
    plt.show()
