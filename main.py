import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch._dynamo
import time
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='GPU Test Pytorch')

# Add arguments
parser.add_argument('test_duration', type=int, help='Test duration in sec', default=3600)

args = parser.parse_args()

torch._dynamo.config.suppress_errors = True

model = models.vgg16().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001)
compiled_model = model

# test max memory size
batch = 64
grow = 4
margin = 0.15
for i in range(1000):
    try: 
        x = torch.randn(batch, 3, 224, 224).cuda()
        optimizer.zero_grad()
        out = compiled_model(x)
        out.square().sum().backward()
        optimizer.step()
        print(out.mean().cpu().detach().numpy())
        batch += grow
        print("grow batch size to %d" % batch)
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        batch = batch - grow - int(batch * margin)
        batch -= batch % 2
        print("test with batch size: %d" % batch)
        del x
        del out
        break

# empty cache
torch.cuda.synchronize()
torch.cuda.empty_cache()

# start testing
start_time = time.time()
print_time = 0
cnt = 0
while True:
    x = torch.randn(batch, 3, 224, 224).cuda()
    optimizer.zero_grad()
    out = compiled_model(x)
    out.square().sum().backward()
    optimizer.step()
    cnt += 1
    
    cur_time = time.time()
    if cur_time - print_time > 10:
        print("[iter %05d, %.1f%%] iter/sec=%f" % (cnt, 100*(cur_time - start_time) / args.test_duration, 
                                                   cnt / (cur_time - start_time)))
        print_time = cur_time
    if cur_time - start_time > args.test_duration:
        break

print("[iter %05d, %.1f%%] iter/sec=%f" % (cnt, 100*(cur_time - start_time) / args.test_duration, 
                                                   cnt / (cur_time - start_time)))