import re
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    epoch = -1
    data = []

    for line in lines:
        if '[INFO]Epoch' in line:
            epoch_match = re.search(r'\[INFO\]Epoch \[(\d+)\]', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
        elif '[DEBUG]' in line:
            debug_match = re.search(r'loss: ([\d.]+) acc@1: ([\d.]+)', line)
            if debug_match:
                loss = float(debug_match.group(1))
                acc1 = float(debug_match.group(2))
                patch = len([d for d in data if d[0] == epoch])
                data.append((epoch, patch, loss, acc1))
    return data

def organize_by_epoch_patch(data):
    patch_data = {}
    for epoch, patch, loss, acc1 in data:
        patch_data.setdefault(epoch, []).append((patch, loss, acc1))
    return patch_data


log1 = parse_log_file("vit_result.txt")
log2 = parse_log_file("cnn_resnet_result.txt")
log3 = parse_log_file("cnn_googlenet_result.txt")
log4 = parse_log_file("cnn_mobilenet_v2_result.txt")

log1_data = organize_by_epoch_patch(log1)
log2_data = organize_by_epoch_patch(log2)
log3_data = organize_by_epoch_patch(log3)
log4_data = organize_by_epoch_patch(log4)


def flatten_log(data):
    xs, loss, acc1 = [], [], []
    for epoch in sorted(data.keys()):
        for patch, l, a in data[epoch]:
            xs.append(len(xs))
            loss.append(l)
            acc1.append(a)
    return xs, loss, acc1

x1, loss1, acc1 = flatten_log(log1_data)
x2, loss2, acc2 = flatten_log(log2_data)
x3, loss3, acc3 = flatten_log(log3_data)
x4, loss4, acc4 = flatten_log(log4_data)

# Display plots
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(x2, loss2, label="Resnet18")
plt.plot(x3, loss3, label="GoogLeNet")
plt.plot(x4, loss4, label="Mobilenet_V2")
plt.plot(x1, loss1, label="SpikingResformer")
plt.title("Loss over Batches")
plt.xlabel("Batch Index")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x2, acc2, label="Resnet18")
plt.plot(x3, acc3, label="GoogLeNet")
plt.plot(x4, acc4, label="Mobilenet_V2")
plt.plot(x1, acc1, label="SpikingResformer")
plt.title("Acc@1 over Batches")
plt.xlabel("Batch Index")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig(f"comparison.jpg")
plt.tight_layout()
plt.show()