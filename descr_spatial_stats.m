function sift_desc_spatialcntx=descr_spatial_stats(sift_desc, frames) 

sift_desc_spatialcntx = zeros(size(sift_desc, 1) + 3, size(sift_desc, 2));
X = frames(1:2, :)';
MdlES = ExhaustiveSearcher(X);    
for j=1:size(frames, 2)
    r = 12 * 2^frames(3, j); %max 150 pixels
    Y = frames(1:2, j)';
    idx = rangesearch(MdlES,Y,r);
    indices_in_x = idx{1};
    p = numel(indices_in_x) - 1; %discount the j frame
    delta_S = 0;
    delta_theta = 0;
    for k=1:numel(indices_in_x) 
        delta_S = delta_S + abs(frames(3, k) - frames(3, j));
        delta_theta = delta_theta + abs(rad2deg(frames(4, k)) - rad2deg(frames(4, j)));
    end
    delta_S = (delta_S / p) * 100;
    delta_theta = ((delta_theta / p) / 360) * 255;
    rows = size(sift_desc, 1);
    sift_desc_spatialcntx(1:rows, j) = sift_desc(:, j);
    if p ~= 0
        sift_desc_spatialcntx(rows+1:rows+3, j) = [p; delta_S; delta_theta];
    end 
end
sift_desc_spatialcntx = uint8(sift_desc_spatialcntx);