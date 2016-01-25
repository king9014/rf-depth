trainset = randperm(1449,795);
trainset = sort(trainset);

testset = zeros(1,1449-795); t=1; p=1;
for i=1:1449
    if p<796 && i~=trainset(p), testset(t)=i; t=t+1;
    else p=p+1; end
end

save('nyuTrainTest.mat','trainset','testset');