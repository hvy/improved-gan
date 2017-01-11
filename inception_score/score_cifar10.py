import chainer
from model import get_inception_score


def chainerim_to_tfim (im):
    return im.transpose((1, 2, 0))


def test_score_cifar10():
    x, _ = chainer.datasets.get_cifar10(ndim=3, withlabel=False)
    x *= 255.0

    ims = []
    # samples = 50000
    samples = 100
    for i in range(samples):
        im = x[i]
        im = chainerim_to_tfim(im)
        ims.append(im)

    mean, std = get_inception_score(ims)

    print('Score mean:', mean)
    print('Score std:', std)


if __name__ == '__main__':
    test_score_cifar10()
