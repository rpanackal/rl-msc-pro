

import torch


def test_unfolding():
    a = torch.arange(15).reshape(1, 5, 3)
    print(a.shape)
    print(a)

    b = a.unfold(1, 2, 1).permute(0, 1, 3, 2)
    print(b.shape)
    print(b)


    c = torch.arange(10).reshape(10, 1)
    print(c)

    d = c.view(5, 2, 1)
    print(d.shape)
    print(d)

    # e = d.view(5, 2, 1)
    # print(e.shape)
    # print(e)

test_unfolding()