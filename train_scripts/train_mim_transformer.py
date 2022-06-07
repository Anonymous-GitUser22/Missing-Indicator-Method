# Code is based on the official SAINT github repository
# https://github.com/somepago/saint

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler
from time import time

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from models.nn_models import Tab_Transformer
from data.nn_data import DataSet
from utils.tabular_utils import get_data
from utils.nn_utils import count_parameters, classification_scores, save_dataset_dict, save_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--optimizer', default='AdamW', type=str, choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str, choices = ['cosine','linear'])
parser.add_argument('--savemodelroot', default='./output', type=str)
parser.add_argument('--run_name', default='transformer', type=str)
parser.add_argument('--seed', default=10 , type=int)

parser.add_argument('--data', default='openml', type=str, choices = ['openml','synthetic'])
parser.add_argument('--dset_name', type=str)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--num_features', default=4, type=int)

parser.add_argument('--miss_mech', default='mnar', type=str, choices = ['mcar','mnar'])
parser.add_argument('--mcar_p', default=0.5, type=float)
parser.add_argument('--mnar_gamma', default=1., type=float)

parser.add_argument('--embedding_size', default=32, type=int)
parser.add_argument('--transformer_depth', default=1, type=int)
parser.add_argument('--attention_heads', default=8, type=int)
parser.add_argument('--viz', default=False, type=bool)
parser.add_argument('--eval_loop', default=False, type=bool)

args = parser.parse_args()

print(f'embedding_size: {args.embedding_size}')
print(f'transformer_depth: {args.transformer_depth}')
print(f'attention_heads: {args.attention_heads}')

if args.data == 'openml':
    modelsave_path = os.path.join(os.getcwd(),args.savemodelroot,args.data,args.dset_name,args.run_name)
elif args.data == 'synthetic':
    modelsave_path = os.path.join(os.getcwd(),args.savemodelroot,args.data+'_'+str(args.seed),args.run_name)
os.makedirs(modelsave_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

X_train, X_test, y_train, y_test = get_data(args)

classes=len(np.unique(y_train))

scalar = StandardScaler()
train_ds = DataSet(X_train, y_train, args.miss_mech, args.seed, args.mcar_p, args.mnar_gamma, scalar, train=True)
trainloader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)

test_ds = DataSet(X_test, y_test, args.miss_mech, args.seed, args.mcar_p, args.mnar_gamma, train_ds.scalar, train=False)
testloader = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False)

# Save datasets to view after training
# save_dataset_dict(train_ds.__dict__, modelsave_path, 'train')
# save_dataset_dict(valid_ds.__dict__, modelsave_path, 'val')
# save_dataset(valid_ds, modelsave_path, 'val')

torch.manual_seed(args.seed)
mlpfory_layers = [args.embedding_size, classes]
if args.viz:
    model = Tab_Transformer(n_in = X_train.shape[-1]*2, n_embed = args.embedding_size, atn_heads = args.attention_heads, \
                        depth=args.transformer_depth, mlpfory_layers = mlpfory_layers, viz_path=modelsave_path)
else:
    model = Tab_Transformer(n_in = X_train.shape[-1]*2, n_embed = args.embedding_size, atn_heads = args.attention_heads, \
                        depth=args.transformer_depth, mlpfory_layers = mlpfory_layers)

model.to(device)

criterion = nn.CrossEntropyLoss().to(device)

if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils.nn_utils import get_scheduler
    scheduler = get_scheduler(args, optimizer)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
elif args.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(),lr=args.lr)

best_valid_auroc = 0
best_valid_accuracy = 0
print('Setup complete, starting training loop')
start_time = time()
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        # cls is the CLS Token, x is our main data, y_gts is our label
        cls, x, y_gts = data[0].to(device), data[1].float().to(device), data[2].long().to(device)

        # We are converting the data to embeddings in the next step
        n1,n2 = x.shape
        x_embedded = torch.empty(n1, n2, args.embedding_size)
        for i in range(n2):
            x_embedded[:,i,:] = model.embeds[i](x[:,i])
        x_embedded = x_embedded.to(device)
        # Adding embedded CLS Token to the start (convention to add to start?)
        cls_embedded = model.cat_embeds(cls)
        x_embedded = torch.cat((cls_embedded,x_embedded),dim=1)

        reps = model.transformer(x_embedded)
        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:,0,:]
        y_outs = model.mlpfory(y_reps)

        loss = criterion(y_outs,y_gts.squeeze()) 
        loss.backward()
        optimizer.step()
        if args.optimizer == 'SGD':
            scheduler.step()
        running_loss += loss.item()
    print(f'Running Minibath Loss: {running_loss}')
    if epoch==0 or (epoch+1)%5==0:
            model.eval()
            with torch.no_grad():
                accuracy, auroc = classification_scores(model, testloader, device, 'transformer', classes)
                print('[EPOCH %d] ACCURACY: %.3f, AUROC: %.3f' %
                        (epoch + 1, accuracy,auroc ))
                # torch.save(model.state_dict(),f'{modelsave_path}/model_epoch_{epoch + 1}.pth')
                if accuracy > best_valid_accuracy:
                    best_valid_accuracy = accuracy
                    best_valid_auroc = auroc
                    # torch.save(model.state_dict(),'%s/bestmodel_viz.pth' % (modelsave_path))
            model.train()

# model.eval()
# with torch.no_grad():
#     accuracy, aoc = classification_scores(model, testloader, device, 'transformer')

# with open(os.path.join(modelsave_path,f'log_{args.seed}.txt'), "w") as f:
#     f.write(f'Accuracy: {accuracy}')

print(f'Total training time: {time()-start_time}')
total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
# print('VALID ACCURACY on best model:  %.3f' %(best_valid_accuracy))
# print('VALID AUROC on best model:  %.3f' %(best_valid_auroc))
print('Final ACCURACY:  %.3f' %(accuracy))

