import torch


def fast_gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack on the input x.
    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is the number of classes.
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels, and N is the image
       dimension.
       The input batch of images. Note that x.requires_grad must have been active before computing the logits
       (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation (before clipping) will have a norm of
        exactly epsilon as measured by the desired norm (see argument: norm).
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", the perturbation (before clipping)
         will have a L_1 norm of exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.

    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

##########################################################
    # YOUR CODE HERE
    loss_fn(logits, y).backward()
    x_pert = torch.Tensor(x.shape)

    if(norm == "inf"):
        # print('linf')
        # implementation based on Eq. (1) https://arxiv.org/pdf/1705.03387.pdf
        # adversarial perturbation = epsilon * sign(gradient)
        perturbed_image = x + epsilon * x.grad.data.sign()
        x_pert = torch.clamp(perturbed_image, 0, 1.0) # return grey scale image, if x < 0 return 0; if x > 1 return 1; else return x

    if(norm == "2"):
        # print('l2')
        # implementation based on Eq. (3) https://arxiv.org/pdf/1705.03387.pdf
        # adversarial perturbation = epsilon * gradient/||gradient||_2
        size = x.shape[1] * x.shape[2] * x.shape[3]
        # simple implementation
        # epsilon = epsilon / torch.sqrt(torch.FloatTensor([size])).item()
        # perturbed_image = x + epsilon * x.grad.data.sign()
        norm_2 = x.grad.data.square().sum(dim = (1,2,3)).sqrt()
        norm_2 = norm_2 * torch.ones([x.shape[0], size]).t()
        norm_2 = norm_2.t().reshape(x.shape)
        perturbed_image = x + epsilon * x.grad.data / norm_2
        x_pert = torch.clamp(perturbed_image, 0, 1.0)
    ##########################################################

    return x_pert.detach()
