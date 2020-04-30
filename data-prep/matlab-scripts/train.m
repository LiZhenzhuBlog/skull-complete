data_path='/Users/shinaushin/Documents/MATLAB/JHU/Spring 2020/Research/skull-complete/data-prep/matlab-scripts/MATs/30';

dest_path='/Users/shinaushin/Documents/MATLAB/JHU/Spring 2020/Research/skull-complete/data-prep/matlab-scripts/MATs';

mat_files=dir(fullfile(data_path, '*.mat'));

for i=1 : length(mat_files)

	if strcmp(mat_files(i).name, '.') || strcmp(mat_files(i).name, '..') || mat_files(i).isdir == 1 || ~strcmp(mat_files(i).name(end-2:end), 'mat')
		continue;
    end
	
	filename = [data_path '\' mat_files(i).name];

	load(filename);

	dataset(i, :,:,:) = defected;

	labels(i,:,:,:) = instance;
	
	destname = [dest_path '/data30.mat'];
	save(destname, 'dataset', 'labels');
end
