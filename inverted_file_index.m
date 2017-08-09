function [invfindex, img_norms, node_weights, dbImgNames]=inverted_file_index(db_dir, vocabulary_tree, improvement)


%1. load each image into the directory
%2. compute features and extract descriptor
%3. compute the path of the descriptors down to the leaves
%4. record for each leaf which images had features occurring at that 
%vocabulary node, and the frequency of occurrence (mi)
%5. Generate the inverted file index only at the leaves: invfindex

%List of images
dbImgNames = dir([db_dir '/*.jpg']);
Nimgs = length(dbImgNames);
assert(Nimgs > 0);
fprintf('%d images\n', Nimgs);

%Inverted file index: this is an array in the size of the number of leaves 
%in the tree
%Each element of the array is a struct with two fields:
% - invfindex(i).images: the images containing the visual word correponding 
%to the leaf
% - invfindex(i).mi: the score which the visual word will vote towards 
%the corresponding image
K = vocabulary_tree.K;
L = vocabulary_tree.depth;
nleaves=K^L;

%inverted file 
invfindex(nleaves).images=[]; %DB image id (index in dbImgNames)
invfindex(nleaves).mi=[]; %each score represents mi for some leaf node i

img_norms = zeros(1, Nimgs);
nnodes = ((K^(L + 1) - 1) / (K - 1)) - 1;
%node_weights = zeros(1, nnodes);
node_Ni = zeros(2, nnodes); %second row holds id of last img seen
%last node number of second last level of nodes (+1 for first leaf node)
last_seclastlvl = nnodes - nleaves; %node before first node on last level

% Extract features for all images and fill in invfindex
for i=1:Nimgs %for each img compute mi
    fprintf('Inverse file index: processing image %d, %s\n', i, dbImgNames(i).name);

    % load the image
    img=imread(fullfile(db_dir,dbImgNames(i).name));
    img=single(rgb2gray(img));
    
    %compute the keypoints and descriptors for this image
    if improvement == 0 %no improvement, DoG, SIFT
        [~, sift_desc] = vl_sift(img);
    elseif improvement == 1 %Affine Invariant Features, SIFT
        [~, sift_desc] = vl_covdet(img, 'Method', 'HarrisLaplace',  'EstimateAffineShape', true);
        sift_desc = sift_desc * 255;
        sift_desc = uint8(sift_desc);
    else %improvement = 2, descriptors with spatial context features
        [frames, sift_desc] = vl_sift(img);
        sift_desc=descr_spatial_stats(sift_desc, frames);
    end
    
    number_of_descriptors = size(sift_desc,2);
    
    %Generate descriptor path down the vocabulary tree and update
    %Ni, and inverted files of leaves containing mi frequency scores.
    for j=1:number_of_descriptors % for each descriptor in image i
        
        sift_desc_j = sift_desc(:, j); %get one sift descriptor      
        path_to_leaf=descr_to_path(sift_desc_j, vocabulary_tree, L);

        index=0;
        for p=1:numel(path_to_leaf) %numel is number of array elements
            index= index*K+path_to_leaf(p); %node along the path
            if node_Ni(2, index) ~= i %last seen img at this node
                node_Ni(1, index) = node_Ni(1, index) + 1;
                node_Ni(2, index) = i;
            end
        end
        
		leaf_index = index - last_seclastlvl;
        image_ind = find(invfindex(leaf_index).images == i); %if image i is already in invfindex(leaf_index).images
        if isempty(image_ind)
            %add the new image (i) to invfindex(index).images if not present with mi score 1
            invfindex(leaf_index).images(end+1) = i; %designate a new mi score
            invfindex(leaf_index).mi(end+1) = 1;
        else
            %increment score by 1 if image i is already in invfindex(index).images
            x = invfindex(leaf_index).mi(image_ind);
            invfindex(leaf_index).mi(image_ind) = x + 1;
        end
    end
end


node_weights = log((Nimgs)./node_Ni(1, :));
node_weights(node_Ni(1, :) == 0) = 0; %how would we deal with NaN?


path_sumOfWeights = zeros(1, nnodes);
%compute sum of node weights for each path in the vocabulary tree
%O(K^L)
for i=1:last_seclastlvl 
    first_child = (i * K) + 1;
    last_child = first_child+K-1;
    if i <= K
        path_sumOfWeights(i) = node_weights(i);
    end
    for j=first_child:last_child
        path_sumOfWeights(j) = path_sumOfWeights(i) + node_weights(j);
    end
end

%path_sumOfWeights(last_seclastlvl+1:nnodes)

%L1 norms = sum_i{leaf_mi*(sum of wi along path_to_leaf)}
%O(K^L * num_imgs)
img_norms = zeros(1, Nimgs); %L1 norm
pindex = last_seclastlvl+1;
for l=1:nleaves %process each leaf
    for im=1:length(invfindex(l).images) 
        id = invfindex(l).images(im);
        cval = img_norms(id);
        img_norms(id) = cval + path_sumOfWeights(pindex)*invfindex(l).mi(im);
    end
    pindex = pindex + 1;
end

