% Read Data
MAX_TRAIN = 40000;
MAX_VAL = 10000;

% data_x = h5read('project_data/train.h5', '/data');
% data_val = h5read('project_data/validate.h5', '/data');

data_x = h5read('train_x_normalized.h5', '/data');
data_y = h5read('project_data/train.h5', '/label');

data_val = h5read('validate_normalized.h5', '/data');

% % Normalize Data
% 
% means = mean(data_x,2);
% stds = std(data_x, 0, 2);
% 
% % means = size(means)
% % stds = size(stds)
% 
% for i = 1:2048
%     disp(i)
%     for j = 1:MAX_TRAIN
%         data_x(i,j) = (data_x(i,j)-means(i))/stds(i);
%     end
% end
% 
% for i = 1:2048
%     disp(i)
%     for j = 1:MAX_VAL
%         data_val(i,j) = (data_val(i,j)-means(i))/stds(i);
%     end
% end
% 
% % Safe Data
% h5create('train_x_normalized.h5','/data', [2048 40000])
% h5write('train_x_normalized.h5','/data', data_x)
% 
% h5create('validate_normalized.h5','/data', [2048 10000])
% h5write('validate_normalized.h5','/data', data_val)
% 
% disp('end of normalizing');


% Training Phase

data_y_Ind = zeros(10,40000);

for i = 1:40000
    v = data_y(i);
    data_y_Ind(v+1,i) = 1;
    
end

data_x_size = size(data_x)
data_y_Ind_size = size(data_y_Ind)

net = patternnet(10);

[net, tr] = train(net, data_x, data_y_Ind);
nntraintool

testX = data_x(:,tr.testInd);
testY = data_y_Ind(:,tr.testInd);

predY = net(testX);

predY(:,1:5)
vecY = vec2ind(testY);
vecY(1:5)

result = sum((vec2ind(testY) ~= vec2ind(predY)))/40000

% Prediction Phase

val_pred = net(data_val);
val_pred_ind = vec2ind(val_pred);
val_pred_ind = val_pred_ind -1;
csvwrite('val_pred_nn.txt', transpose(val_pred_ind));


