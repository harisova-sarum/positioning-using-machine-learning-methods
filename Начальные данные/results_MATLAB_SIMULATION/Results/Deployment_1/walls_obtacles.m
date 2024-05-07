clc
clear all

load('cir.mat')
load('RxPos.mat')

DATA_FULL_Tx2_WALLS_EMPTY = {cir.data{1,2},RxPos};
DATA_FULL_Tx3_WALLS_EMPTY = {cir.data{1,3},RxPos};

Matrix = DATA_FULL_Tx3_WALLS_EMPTY;
A = zeros(561,15*7);

for i = 1 : 561
    if isempty(Matrix{1,1}{i,1}) == 1
        continue
    elseif size(Matrix{1,1}{i,1},1) < 15
        s = size(Matrix{1,1}{i,1},1);
        A(i,:) = [reshape(Matrix{1,1}{i,1}', ...
            1,s*7) zeros(1,(15 - s)*7)];
    else
        A(i,:) = reshape(Matrix{1,1}{i,1}',1,15*7);
    end
end

B = [A Matrix{1,2}];

csvwrite('DATA_FULL_Tx3_WALLS_EMPTY.csv',B)