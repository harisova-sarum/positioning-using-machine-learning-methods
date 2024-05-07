
%load('C:\Users\Азалия\Desktop\Диплом\Начальные данные\results_MATLAB_SIMULATION\Results\Deployment_4\DATA_FULL_Tx3_WALLS_OBTACLES.mat')

A = zeros(561,15*7);

for i = 1 : 561
    if isempty(DATA_FULL_Tx3_WALLS_OBTACLES{1,1}{i,1}) == 1
        continue
    elseif size(DATA_FULL_Tx3_WALLS_OBTACLES{1,1}{i,1},1) < 15
        s = size(DATA_FULL_Tx3_WALLS_OBTACLES{1,1}{i,1},1);
        A(i,:) = [reshape(DATA_FULL_Tx3_WALLS_OBTACLES{1,1}{i,1}', ...
            1,s*7) zeros(1,(15 - s)*7)];
    else
        A(i,:) = reshape(DATA_FULL_Tx3_WALLS_OBTACLES{1,1}{i,1}',1,15*7);
    end
end

B = [A DATA_FULL_Tx3_WALLS_OBTACLES{1,2}];
csvwrite('DATA_FULL_Tx3_WALLS_OBTACLES.csv',B)