function generate_LR_metasr_X1_X4()
%% settings
path_save = './B100';
path_src = './B100/HR';
ext               =  {'*.png'};
filepaths           =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(path_src, ext{i})));
end
nb_im = length(filepaths);
DIV2K_HR = [];

for idx_im = 1:nb_im
    fprintf('Read HR :%d\n', idx_im);
    ImHR = imread(fullfile(path_src, filepaths(idx_im).name));
    DIV2K_HR{idx_im} = ImHR;
end
FolderLR_bicubic = fullfile(path_save,'LR_bicubic')
if ~exist(FolderLR_bicubic)
    mkdir(FolderLR_bicubic)
end
    
%% generate and save LR via imresize() with Bicubic
for scale=1.1:0.1:4
    FolderLR = fullfile(path_save,'LR_bicubic', sprintf('X%.2f',scale));
    
    if ~exist(FolderLR)
        mkdir(FolderLR)
    end
    for IdxIm = 1:nb_im
        fprintf('IdxIm=%d\n', IdxIm);
        ImHR = DIV2K_HR{IdxIm};
        [h,w,n]=size(ImHR);
        if mod(h,scale) ~= 0
            h = h - 4;
        end
        if mod(w,scale) ~= 0
            w = w - 4;
        end
        ImHR =ImHR(1:h,1:w,:);
        ImLR = imresize(ImHR, 1/scale, 'bicubic');
        % name image
        fileName = filepaths(IdxIm).name
        NameLR = fullfile(FolderLR, fileName);
        % save image
        imwrite(ImLR, NameLR, 'png');
    end
end


end
