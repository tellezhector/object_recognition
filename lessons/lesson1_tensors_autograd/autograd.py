import torch

w_true = 2.0
b_true = -1.0

# small, so the line is mostly visible through it
noise = torch.randn(100) * 0.5
x = torch.randn(100)
# the target data — fixed relationship + a little noise
y = w_true * x + b_true + noise

print(f"{noise=}")
print(f"{x=}")
print(f"{y=}")

w = (torch.rand(1) * 10 - 5).requires_grad_()
b = (torch.rand(1) * 10 - 5).requires_grad_()


learning_rate = 0.1
epochs = 200

for epoch in range(epochs):
    y_pred = w * x + b
    # mse = minimal square error
    mse = ((y_pred - y) ** 2).mean()

    # computes
    # d(mse)/dw and d(mse)/db,
    # storing them in w.grad and b.grad
    mse.backward()

    # w and b require_grad=True, so any operation on them normally gets tracked
    # for autograd. We don't want that here — this is a plain in-place value
    # update, not part of the loss computation, so no_grad() tells PyTorch to
    # skip building graph history for what happens inside this block.
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # backward() ADDS the new gradients onto whatever's already in .grad rather
    # than replacing them. Without zeroing here, next iteration's gradients
    # would stack on top of this one's, making the updates wrong.
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:3d}  mse={mse.item():.4f}  w={w.item():.3f}  b={b.item():.3f}")

print(f"\nlearned: w={w.item():.3f} (true {w_true}), b={b.item():.3f} (true {b_true})")
