function y=nanmean(x)

p=isnan(x);
x(p)=0;
y=sum(x(:))/sum(~p(:));%sum(~p(:)):p中为0元素的个数总和，也就是x中不是nan的个数
                       %sum(x(:))：x中除了nan的的元素的累加和
                       %y的意义是x中非nan总和除以非nan个数
                       
                       %现在的问题是x中除了nan就是0%%%%%%%%%%%%%%%%%%%%%%%%%