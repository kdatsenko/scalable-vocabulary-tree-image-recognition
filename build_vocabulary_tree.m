function vocabulary_tree = build_vocabulary_tree(db_dir, k, L, improvement)

branching_factor = k;
vocab_size = k^L;

dbImgNames = dir([db_dir '/*.jpg']);
all_sift_desc = [];

NImgs = length(dbImgNames);
assert(NImgs > 0);
fprintf('%d images\n', NImgs);
for i = 1:NImgs
    fprintf('Vocabulary tree: processing image %d, %s\n', i, dbImgNames(i).name);

    %DO FOR ALL IMAGS
    I=imread(fullfile(db_dir,dbImgNames(i).name));
    img = single(rgb2gray(I));

    %compute the keypoints and descriptors
    if improvement == 0 %no improvement, DoG, SIFT
        [~, sift_desc] = vl_sift(img);
    elseif improvement == 1 %Affine Invariant Features, SIFT
        [~, sift_desc] = vl_covdet(img, 'Method', 'HarrisLaplace',  'EstimateAffineShape', true);
        sift_desc = sift_desc * 255;
    else %improvement = 2, descriptors with spatial context features
        [frames, sift_desc] = vl_sift(img);
        sift_desc=descr_spatial_stats(sift_desc, frames);
    end    
    
    if isempty(all_sift_desc)
        all_sift_desc = sift_desc;
    else
        %gather all sift descriptors together
        if i > 1
            all_sift_desc = [all_sift_desc sift_desc];
        end
    end
end

fprintf('Number of extracted features: %d\n', size(all_sift_desc, 2));

% Cluster the features Hierarchical K-Means
fprintf('Building Vocabulary tree...\n');
%Compute the visual words and the vocabulary tree using the function vl_hikmeans:
vocabulary_tree = vl_hikmeans(uint8(all_sift_desc), branching_factor, vocab_size);
