%% STEP 1: Construct the de-enhanced images

%% pre-define the target curve y and the dictionary 

% DCE_4D_original_data is the 4D original DCE-MRI, whose size is (mx, my, mz, timepoints)
% target_curve is the estimated ideal time-intensity curve in the manually
% outlined Liver ROI, whose coordinates is (coordinates_x,coordinates_y,coordinates_z)
coordinates_x = 175; coordinates_y = 145; coordinates_z = 10;

for i=1:22
    DCE_4D_original_data(:,:,:,i)=y_Read(['/Volumes/TR-REG/arraged_liver_data/patient2/3D_ORIGINAL_SLICE/extend_slice_',num2str(i),'.nii']);
end

target_curve=mean(mean(mean(DCE_4D_original_data(coordinates_x-1:coordinates_x+1,coordinates_y-1:coordinates_y+1,...
    coordinates_z-1:coordinates_z+1,:))));

Dictionary_composed=reshape(size(DCE_4D_original_data,1)*size(DCE_4D_original_data,2)*sizesize(DCE_4D_original_data,3),[size(DCE_4D_original_data,1)*size(DCE_4D_original_data,2)*sizesize(DCE_4D_original_data,3),size(DCE_4D_original_data,4)]);
Dictionary_composed=Dictionary_composed';
%% solve the coefficients x 
% When target_curve and Dictionary_composed are already defined, we can use
% SLEP toolbox to solve the coefficients x
% SLEP parameters definiation
opts.G=[1,2:size(DCE_4D_original_data,1)*size(DCE_4D_original_data,2)*sizesize(DCE_4D_original_data,3)];
opts.ind=[[1, 1, 1]',[1+1, size(DCE_4D_original_data,1)*size(DCE_4D_original_data,2)*sizesize(DCE_4D_original_data,3), 1]'];
opts.init=2;        % starting from a zero point
opts.tFlag=5;       % the relative change is less than opts.tol
opts.maxIter=1000;  % maximum number of iterations
opts.tol=1e-5;      % the tolerance parameter
opts.rFlag=1;       % use ratio
opts.nFlag=0;       % without normalization
z=[0.1,0.1];

% correlation-weighted constraints
correlation_coefficients=zeros(size(DCE_4D_original_data,1),size(DCE_4D_original_data,2),size(DCE_4D_original_data,3));
for i=1:size(DCE_4D_original_data,1)
    for j=1:size(DCE_4D_original_data,2)
        for k=1:size(DCE_4D_original_data,3)
            correlation_coefficients(i,j,k)=corr(target_curve,squeeze(DCE_4D_original_data(i,j,k,:)));
        end
    end
end
sigma=0.5;
correlation_coefficients(isnan(correlation_coefficients)==1)=0;
correlation_coefficients(correlation_coefficients<0)=0;
correlation_constraints=exp(-(correlation_coefficients(:).^2/sigma));
% solve the coefficients x
[x, funVal, ValueL]= sgLeastR_yj(Dictionary_composed, target_curve, z, opts, correlation_constraints);
% reshape to coefficients map
coefficients_map=reshape(x,[size(DCE_4D_original_data,1),size(DCE_4D_original_data,2),size(DCE_4D_original_data,3)]);

%% compute de-enhanced frames
de_enhanced_images=zeros(size(DCE_4D_original_data));
% pre_contrast_timepoints is the number of pre-contrast frames
de_enhanced_images(:,:,:,1:pre_contrast_timepoints)=DCE_4D_original_data(:,:,:,1:pre_contrast_timepoints);
for timepoints=pre_contrast_timepoints:size(DCE_4D_original_data,4)
    ratio=((DCE_4D_original_data(coordinates_x, coordinates_y, coordinates_z, timepoints))-(DCE_4D_original_data(coordinates_x,coordinates_y,coordinates_z,1)))...
        ./(coefficients_map(coordinates_x,coordinates_y,coordinates_z));
    de_enhanced_images(:,:,:,timepoints)=abs(DCE_4D_original_data(:,:,:,timepoints)-coefficients_map.*ratio);
end

    header.dim=[size_x(patient_no),size_x(patient_no),extend_slice];header.dt=[16 0];
    final=zeros(size_x(patient_no),size_x(patient_no),extend_slice,Timepoints);final(:,:,ceil((extend_slice-22)/2)+1:ceil((extend_slice-22)/2)+22,:)=jii;
    
    for ii=1:Timepoints;
        rest_WriteNiftiImage(final(:,:,:,ii),header,[filepath,'yj_new_group/no_graphcut_second_sub_task_0_1_aorta_mean_98_group_5_5_sigma_50_',num2str(ii),'.nii']);
    end
    
%% STEP 2: RC registration with MIRT toolbox   
    
% let the frame with the highest concentrate contrast as the reference
refim=de_enhanced_images(:,:,:,10);
refim=(refim-min(min(min(refim))))./(max(max(max(refim)))-min(min(min(refim))));

% parameters settings for MIRT toolbox

main.similarity='rc';   % similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
main.alpha=0.05;        % similarity measure parameter (e.g., alpha of RC)
main.subdivide=3;       % use 3 hierarchical levels
main.okno=4;            % mesh window size, the smaller it is the more complex deformations are possible
main.lambda = 0.1;    % transformation regularization weight, 0 for none
main.single=0;          % show mesh transformation at every iteration

%Optimization settings
optim.maxsteps = 300;   % maximum number of iterations at each hierarchical level
optim.fundif = 1e-7;    % tolerance (stopping criterion)
optim.gamma =1;       % initial optimization step size
optim.anneal=0.8;       % annealing rate on the optimization step

final_registered_images = zeros(size(DCE_4D_original_data));
for timepoints=1:size(DCE_4D_original_data,4)
    if timepoints~=10
        im=de_enhanced_images(:,:,:,timepoints);
        im=(im-min(min(min(im))))./(max(max(max(im)))-min(min(min(im))));
        [res, ~]=mirt3D_register(refim, im, main, optim);
        [Xx,Xy,Xz]=mirt3D_nodes2grid(res.X, main.F, main.okno);
        deformation_field=cat(4,Xx,Xy,Xz);
        final_registered_images(:,:,:,timepoints) = mirt3D_mexinterp(DCE_4D_original_data(:,:,:,timepoints), Xx, Xy, Xz);
    else
        final_registered_images(:,:,:,timepoints) = DCE_4D_original_data(:,:,:,timepoints);
    end
end


