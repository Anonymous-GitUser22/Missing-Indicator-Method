# Code is based on the official SAINT github repository
# https://github.com/somepago/saint

from modulefinder import packagePathMap
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

from models.nn_models import simple_MLP
from data.nn_data import DataSet
from utils.tabular_utils import get_data
from utils.nn_utils import count_parameters, classification_scores

parser = argparse.ArgumentParser()

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--optimizer', default='AdamW', type=str, choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str, choices = ['cosine','linear'])
parser.add_argument('--savemodelroot', default='./output', type=str)
parser.add_argument('--run_name', default='mlp', type=str)
parser.add_argument('--seed', default=10 , type=int)

parser.add_argument('--data', default='openml', type=str, choices = ['openml','synthetic'])
parser.add_argument('--dset_name',type=str)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--num_features', default=4, type=int)

parser.add_argument('--miss_mech', default='mnar', type=str, choices = ['mcar','mnar'])
parser.add_argument('--mcar_p', default=0.5, type=float)
parser.add_argument('--mnar_gamma', default=1., type=float)

parser.add_argument('--mlp_layers', default=['8p', '4p'], nargs='+', type=str)
parser.add_argument('--eval_loop', default=False, type=bool)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Device is {device}.")

if args.data == 'openml':
    modelsave_path = os.path.join(os.getcwd(),args.savemodelroot,args.data,args.dset_name,args.run_name)
elif args.data == 'synthetic':
    modelsave_path = os.path.join(os.getcwd(),args.savemodelroot, \
                            args.data+'_'+str(args.seed),args.run_name+'_'+str(args.mlp_layers))
os.makedirs(modelsave_path, exist_ok=True)

X_train, X_test, y_train, y_test = get_data(args)

classes=len(np.unique(y_train))

scalar = StandardScaler()
train_ds = DataSet(X_train, y_train, args.miss_mech, args.seed, args.mcar_p, args.mnar_gamma, scalar, train=True)
trainloader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True)

test_ds = DataSet(X_test, y_test, args.miss_mech, args.seed, args.mcar_p, args.mnar_gamma, train_ds.scalar, train=False)
testloader = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False)

# MLP layers
p = X_train.shape[-1]
mlp_layers = [2*p]
for l in args.mlp_layers:
    if 'p' in l:
        l = int(l[:-1])*p
    else:
        l = int(l)
    mlp_layers.append(l)
mlp_layers.append(classes)
print(f'Mlp layers: {mlp_layers}')

torch.manual_seed(args.seed)
model = simple_MLP(mlp_layers)

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

# print('Setup complete, starting training loop')
best_valid_auroc = 0
best_valid_accuracy = 0
start_time = time()
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()

        # x is our main data, y_gts is our label (no cls for mlp training)
        _, x, y_gts = data[0].to(device), data[1].float().to(device), data[2].long().to(device)

        y_outs = model(x)

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
                accuracy, auroc = classification_scores(model, testloader, device, 'mlp', classes)
                print('[EPOCH %d] ACCURACY: %.3f, AUROC: %.3f' %
                        (epoch + 1, accuracy,auroc ))
                # torch.save(model.state_dict(),f'{modelsave_path}/model_epoch_{epoch + 1}.pth')
                if accuracy > best_valid_accuracy:
                    best_valid_accuracy = accuracy
                    best_valid_auroc = auroc
                    # torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
            model.train()

# model.eval()
# with torch.no_grad():
#     accuracy, _ = classification_scores(model, testloader, device, 'mlp')

# with open(os.path.join(modelsave_path,f'log_{args.seed}.txt'), "w") as f:
#     f.write(f'Accuracy: {accuracy}')

print(f'Total training time: {time()-start_time}')
total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
# print('VALID ACCURACY on best model:  %.3f' %(best_valid_accuracy))
# print('VALID AUROC on best model:  %.3f' %(best_valid_auroc))
print('Final ACCURACY:  %.3f' %(accuracy))