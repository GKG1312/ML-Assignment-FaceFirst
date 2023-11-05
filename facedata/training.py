# def train_model_2(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, earlyStopping=None, history=None):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)

#     dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

#     best_model_wts = copy.deepcopy(model.state_dict())

#     if history is None:
#         history = {'train_loss': [],  'val_loss': [], 'best_epoch': 0,'val_loss_best': 1e10,  'prev_epochs': 0}

#     best_loss = history['val_loss_best']
#     prev_epochs = history['prev_epochs']

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch+1}/{num_epochs}')
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0
#             running_size = 0

#             # Iterate over data.
#             with tqdm(dataloaders[phase], unit="batch") as t:
#                 t.set_description("Training" if (phase == 'train') else "Validation")
#                 # Loop over batches
#                 for inputs, labels in t:
#                     inputs = list(image.to(device) for image in inputs)
#                     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
#                     if phase=='train':
#                         # zero the parameter gradients
#                         optimizer.zero_grad()
#                         loss_dict = model([inputs[0]], [targets[0]])
#                         losses = sum(loss for loss in loss_dict.values())
#                         loss_value = losses.item()
#                         losses.backward()
#                         optimizer.step()
#                     else:
#                         with torch.no_grad():
#                             loss_dict = model([inputs[0]], [targets[0]])
#                         losses = sum(loss for loss in loss_dict.values())
#                         loss_value = losses.item()

#                     epoch_loss = loss_value

#                     t.set_postfix(loss=f"{epoch_loss:.4f}")

#             if phase == 'train':
#                 history['train_loss'].append(epoch_loss)

#             if phase == 'val':
#                 history['val_loss'].append(epoch_loss)

        
#         scheduler.step(history['val_loss'][-1])

#         # After train & val deep copy the best model
#         if epoch_loss < best_loss:
#             best_loss = epoch_loss
#             history['best_epoch'] = epoch + prev_epochs + 1
#             history['val_loss_best'] = epoch_loss
#             best_model_wts = copy.deepcopy(model.state_dict())

#         # Check early stopping
#         if earlyStopping is not None:
#             if earlyStopping.step(history):
#                 print("Early stopping")
#                 break

#         print()

#     print(f'Best Validation loss: {best_loss:4f}')
#     history['prev_epochs'] = epoch + prev_epochs + 1

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model, history



