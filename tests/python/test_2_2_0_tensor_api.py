import math
import venlacpu


def make_tensor(values, shape):
    tensor = venlacpu.Tensor.zeros(
        venlacpu.Shape(shape),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    data = tensor.data_as_float() if hasattr(
        tensor,
        "data_as_float",
    ) else None

    return tensor


def test_version_3_1_1():
    assert venlacpu.__version__ == "3.1.1"


def test_scalar_add():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([4]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    assert (x + 2).tolist() == [
        3.0,
        3.0,
        3.0,
        3.0,
    ]

    assert (2 + x).tolist() == [
        3.0,
        3.0,
        3.0,
        3.0,
    ]


def test_scalar_sub():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([4]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    assert (x - 2).tolist() == [
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ]

    assert (2 - x).tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
    ]


def test_scalar_mul():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([4]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    assert (x * 3).tolist() == [
        3.0,
        3.0,
        3.0,
        3.0,
    ]

    assert (3 * x).tolist() == [
        3.0,
        3.0,
        3.0,
        3.0,
    ]


def test_scalar_div():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([4]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    assert (x / 2).tolist() == [
        0.5,
        0.5,
        0.5,
        0.5,
    ]

    assert (2 / x).tolist() == [
        2.0,
        2.0,
        2.0,
        2.0,
    ]


def test_item():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    assert math.isclose(
        x.item(),
        1.0,
    )


def test_reshape_method():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([2, 3]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    y = x.reshape(
        venlacpu.Shape([3, 2])
    )

    assert y.shape == (
        3,
        2,
    )

    assert y.numel == 6


def test_flatten_method():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([2, 3, 4]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    y = x.flatten()

    assert y.shape == (
        24,
    )


def test_unsqueeze_squeeze():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([2, 3]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    y = x.unsqueeze(1)

    assert y.shape == (
        2,
        1,
        3,
    )

    z = y.squeeze(1)

    assert z.shape == (
        2,
        3,
    )


def test_broadcasting():
    a = venlacpu.Tensor.ones(
        venlacpu.Shape([2, 3]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    b = venlacpu.Tensor.ones(
        venlacpu.Shape([3]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    c = a + b

    assert c.shape == (
        2,
        3,
    )

    assert c.tolist() == [
        [
            2.0,
            2.0,
            2.0,
        ],
        [
            2.0,
            2.0,
            2.0,
        ],
    ]


def test_autograd_scalar():
    x = venlacpu.Tensor.ones(
        venlacpu.Shape([4]),
        venlacpu.DType.Float32,
        venlacpu.Device.cpu(),
    )

    x.requires_grad_(True)

    y = (x * 2).sum()

    y.backward()

    assert x.has_grad()

    gradient = x.grad()

    assert gradient.tolist() == [
        2.0,
        2.0,
        2.0,
        2.0,
    ]
