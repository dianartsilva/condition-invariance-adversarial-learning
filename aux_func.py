import matplotlib.pyplot as plt
from matplotlib import patches, text, patheffects

def save_fig(img, name):
    fig = plt.figure(figsize=(10,5))
    plt.imshow(img)
    plt.show()
    plt.close(fig)
    fig.savefig(f'images/{name}.png',bbox_inches='tight', dpi=150)

def bb_boxes_plot(image, targets, objects, i):
    fig, ax = plt.subplots(figsize=(10,5))

    ax.xaxis.tick_top()
    ax.imshow(image)

    for m in range(len(targets[0]['labels'])):
        ax.add_patch(patches.Rectangle((targets[0]['bboxes'][m][0], targets[0]['bboxes'][m][1]), targets[0]['bboxes'][m][2] - targets[0]['bboxes'][m][0], targets[0]['bboxes'][m][3] - targets[0]['bboxes'][m][1], fill=False, edgecolor='green', lw=2))
        ax.text(targets[0]['bboxes'][m][0], targets[0]['bboxes'][m][1], objects[targets[0]['labels'][m]],verticalalignment='top', color='green',fontsize=10,weight='bold')

    plt.show()
    plt.close(fig)
    fig.savefig(f'DAWN_eval/img_{i}.png',bbox_inches='tight', dpi=150)


#### BBOXES PLOTTING EVALUATION #######

def bboxes_plot_GT(images, targets, colors, objects, dataset, i):
    
    fig, ax = plt.subplots(figsize=(10,5))

    ax.xaxis.tick_top()
    ax.imshow(images.squeeze().permute(1, 2, 0).cpu())

    for m in range(len(targets[0]['labels'])):
        color = colors[targets[0]['labels'][m]-1] 
        ax.add_patch(patches.Rectangle((targets[0]['boxes'][m][0], targets[0]['boxes'][m][1]), targets[0]['boxes'][m][2] - targets[0]['boxes'][m][0], targets[0]['boxes'][m][3] - targets[0]['boxes'][m][1], fill=False, edgecolor=f'{color}', lw=2))
        # ax.text(targets[0]['boxes'][m][0], targets[0]['boxes'][m][1], objects[targets[0]['labels'][m]-1],verticalalignment='top', color='green',fontsize=10,weight='bold')

    plt.title('Ground Truth')
    plt.show()
    plt.close(fig)
    fig.savefig(f'{dataset}_eval/img_{i}_GT.png',bbox_inches='tight', dpi=150)


def bboxes_plot_PRED(images, preds, colors, objects, dataset, i):
    fig, ax = plt.subplots(figsize=(10,5))

    ax.xaxis.tick_top()
    ax.imshow(images.squeeze().permute(1, 2, 0).cpu())

    preds = [{k: v.cpu() for k, v in p.items()} for p in preds]

    for n in range(len(preds[0]['labels'])):
        color = colors[preds[0]['labels'][n]-1] 
        ax.add_patch(patches.Rectangle((preds[0]['boxes'][n][0], preds[0]['boxes'][n][1]), preds[0]['boxes'][n][2] - preds[0]['boxes'][n][0], preds[0]['boxes'][n][3] - preds[0]['boxes'][n][1], fill=False, edgecolor=f'{color}', lw=2, linestyle='--'))
        # ax.text(preds[0]['boxes'][n][0], preds[0]['boxes'][n][1], objects[preds[0]['labels'][n]],verticalalignment='top', color='red',fontsize=10,weight='bold')

    plt.title('Predictions')
    plt.show()
    plt.close(fig)
    fig.savefig(f'{dataset}_eval/img_{i}_PD.png',bbox_inches='tight', dpi=150)


def bboxes_plot_GT_PRED(images, targets, preds, colors, objects, dataset, i):
    
    fig, ax = plt.subplots(figsize=(10,5))

    ax.xaxis.tick_top()
    ax.imshow(images.squeeze().permute(1, 2, 0).cpu())

    for m in range(len(targets[0]['labels'])):
        color = colors[targets[0]['labels'][m]-1] 
        ax.add_patch(patches.Rectangle((targets[0]['boxes'][m][0], targets[0]['boxes'][m][1]), targets[0]['boxes'][m][2] - targets[0]['boxes'][m][0], targets[0]['boxes'][m][3] - targets[0]['boxes'][m][1], fill=False, edgecolor=f'{color}', lw=2))
        # ax.text(targets[0]['boxes'][m][0], targets[0]['boxes'][m][1], objects[targets[0]['labels'][m]-1],verticalalignment='top', color='green',fontsize=10,weight='bold')

    preds = [{k: v.cpu() for k, v in p.items()} for p in preds]

    for n in range(len(preds[0]['labels'])):
        color = colors[preds[0]['labels'][n]-1] 
        ax.add_patch(patches.Rectangle((preds[0]['boxes'][n][0], preds[0]['boxes'][n][1]), preds[0]['boxes'][n][2] - preds[0]['boxes'][n][0], preds[0]['boxes'][n][3] - preds[0]['boxes'][n][1], fill=False, edgecolor=f'{color}', lw=2, linestyle='--'))
        # ax.text(preds[0]['boxes'][n][0], preds[0]['boxes'][n][1], objects[preds[0]['labels'][n]],verticalalignment='top', color='red',fontsize=10,weight='bold')

    plt.title('Ground Truth (full line) + Predictions (dashed)')
    plt.show()
    plt.close(fig)
    fig.savefig(f'{dataset}_eval/img_{i}_GTPred.png',bbox_inches='tight', dpi=150)