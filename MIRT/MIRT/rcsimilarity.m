function [f]=rcsimilarity(R1,R2,alpha)    
        
        rbig=R1-R2;
        
        [y,x]=find_imagebox(rbig); r=rbig(y,x);
        r(isnan(r))=nanmean(r(:));

        Qr=mirt_dctn(r);
        Li=Qr.^2+alpha;

        f=0.5*sum(log(Li(:)/alpha));
       
        
function [y,x]=find_imagebox(im)
    [i,j]=find(~isnan(im)); 
    n=4; % border size
    y=min(i)+n:max(i)-n;
    x=min(j)+n:max(j)-n;
