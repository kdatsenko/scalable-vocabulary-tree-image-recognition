function display_candidates(db_dir, candidates, scores)

dbImgNames = dir(fullfile(db_dir, '*.png'));
if (length(dbImgNames)<=0), dbImgNames = dir(fullfile(db_dir, '*.jpg')); end

figure(2);
clf;

for i = 1:numel(candidates)
    subplot(4, 3, i);
    img=imread(fullfile(db_dir,dbImgNames(candidates(i)).name));
    imagesc(img);
    set(gcf, 'color', 'white');
    axis off;
    title(['Score ',num2str(scores(i))]);
end
