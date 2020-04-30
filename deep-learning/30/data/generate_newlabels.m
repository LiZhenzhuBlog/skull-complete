load('test_data.mat');
load('train_data.mat');

% faulty_data_idx = [235];
% faulty_data_idx = fliplr(faulty_data_idx);
% for i=1:length(faulty_data_idx)
%     dataset(faulty_data_idx(i),:,:,:) = [];
%     labels(faulty_data_idx(i),:,:,:) = [];
% end

newtrain = [];
newlabels = dataset;

num_samples = 1;
for a=1:length(labels)
    defect = labels(a,:,:,:) - dataset(a,:,:,:);
    defect_3dlocs = get3dLocs(defect);
    try
        [k,av] = convhull(defect_3dlocs);
        convhull3d = [defect_3dlocs(k,1), defect_3dlocs(k,2), defect_3dlocs(k,3)];

        contour = [];
        count = 1;
        dataset_3dlocs = get3dLocs(dataset(a,:,:,:));
        for b=1:length(convhull3d)
            nearestIdx = knnsearch(dataset_3dlocs,convhull3d(b,:));
            if isempty(contour) || ~nnz(ismember(contour, dataset_3dlocs(nearestIdx,:), 'rows'))
                contour(count,:) = dataset_3dlocs(nearestIdx,:);
                count = count + 1;
            end
        end

        for b=1:length(contour)
            newlabels(num_samples,contour(b,1),contour(b,2),contour(b,3)) = 2; % label as contour
        end
        newtrain(num_samples,:,:,:) = dataset(a,:,:,:);
        num_samples = num_samples + 1;
    catch ME
        continue;
    end
end

disp(num_samples);
save('train.mat', 'newtrain');
save('labels.mat', 'newlabels');

