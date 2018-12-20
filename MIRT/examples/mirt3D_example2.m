% MIRT3D_EXAMPLE2: Non-rigid 3D registration example 2.
% Brain MRI (T1) image registration. The images are taken from BrainWeb
% www.bic.mni.mcgill.ca/brainweb/
% clear all; close all; clc;
% load DCE_2-1.mat;
% load mirt3D_brain1.mat

% Main settings
main.similarity='RC';  % similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI   
main.subdivide=3;       % use 3 hierarchical levels
main.okno=4;            % mesh window size
main.lambda = 0.1;     % transformation regularization weight, 0 for none
main.alpha=0.05;        % similarity measure parameter (use with RC, MS or CD2)    


% Optimization settings
optim.maxsteps = 30;    % maximum number of iterations at each hierarchical level
optim.fundif = 1e-6;     % tolerance (stopping criterion)
optim.gamma = 1;         % initial optimization step size 
optim.anneal=0.8;        % annealing rate on the optimization step    
 

[res, newim]=mirt3D_register(refim, im, main, optim);

Spacing=[main.okno,main.okno,main.okno];
% [YY,XX,ZZ]=meshgrid(1-main.okno:main.okno:320+2*main.okno, 1-main.okno:main.okno:320+2*main.okno,...
%     1-main.okno:main.okno:22+2*main.okno);
    O(:,:,:,2)=res.X(:,:,:,1);%%%%%%%%%%%
    O(:,:,:,1)=res.X(:,:,:,2);%%%%%%%%%%%
    O(:,:,:,3)=res.X(:,:,:,3);
    Deformatioanfield=O;
[new_Img,T]=bspline_transform(Deformatioanfield,im,Spacing,2);
% res is a structure of resulting transformation parameters
% newim is a deformed image 
%
% you can also apply the resulting transformation directly as
% newim=mirt3D_transform(im, res);

% figure,imshow(refim(:,:,1));impixelinfo; title('Reference (fixed) image slice');
% figure,imshow(im(:,:,1));impixelinfo;    title('Source (float) image slice');
% for i=1:24
%     figure,imshow(newim(:,:,i));impixelinfo; title('Registered (deformed) image slice');
% end
