function off_to_mat(off_path, data_path, volume_size, total_size)

% Put the mesh object in a volume grid and save the volumetric
% represenation file.
% off_path: root off data folder
% data_path: destination volumetric data folder

% total_size = pad_size * 2 + volume_size;

types = ["scalp"; "skull"; "brain"];

dest_path = strcat(data_path, '/', types(1), '_MATs'); % '/' num2str(data_size)];   

if ~exist(dest_path, 'dir')
    mkdir(dest_path);
end     

% for train and test phases
off_list = strcat(off_path, '/', types(1), '/');

dest_tsdf_path = strcat(dest_path, '/');

if ~exist(dest_tsdf_path, 'dir')
    mkdir(dest_tsdf_path);
end

files = dir(fullfile(off_list, '*.off'));
 
for i = 1 : length(files)
    
    max_dims = [0,0,0];
    
    for j = 1 : 3
        dest_path = strcat(data_path, '/', types(j), '_MATs'); % '/' num2str(data_size)];   

        if ~exist(dest_path, 'dir')
            mkdir(dest_path);
        end     

        % for train and test phases
        off_list = strcat(off_path, '/', types(j), '/');

        dest_tsdf_path = strcat(dest_path, '/');

        if ~exist(dest_tsdf_path, 'dir')
            mkdir(dest_tsdf_path);
        end

        files = dir(fullfile(off_list, '*.off'));

        %strcmp = string compare - para assegurar que sao os ficheiros certos
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end           
        filename = strcat(off_list, '/', files(i).name);

        destname = strcat(dest_tsdf_path, '/', files(i).name(1:end-4), '.mat');

        off_data = off_loader(filename);
        lengths = max(off_data.vertices) - min(off_data.vertices);
        max_dims = [max(max_dims(1),lengths(1)), max(max_dims(2),lengths(2)), max(max_dims(3),lengths(3))];
        ratio = lengths ./ max_dims;
        
        if mod(floor(volume_size*ratio(1)),2) == 1
            x = floor(volume_size*ratio(1))-1;
        else
            x = floor(volume_size*ratio(1));
        end
        
        if mod(floor(volume_size*ratio(2)),2) == 1
            y = floor(volume_size*ratio(2))-1;
        else
            y = floor(volume_size*ratio(2));
        end
        
        if mod(floor(volume_size*ratio(3)),2) == 1
            z = floor(volume_size*ratio(3))-1;
        else
            z = floor(volume_size*ratio(3));
        end
        
        instance = polygon2voxel(off_data, [x, y, z], 'auto');
        instance = padarray(instance, [(total_size-x)/2, (total_size-y)/2, (total_size-z)/2]); % pad_size
        instance = int8(instance);
        save(destname, 'instance');
    end
end


function offobj = off_loader(filename, axis, stretch)

% read the off file. The function can also give some initial manipulations
% of the 3D data like rotation and stretch.
% offobj: a struct contains vertices and faces of 3D mesh models.

offobj = struct();
fid = fopen(filename, 'rb');
OFF_sign = fscanf(fid, '%c', 3);
assert(strcmp(OFF_sign, 'OFF') == 1);
 info = fscanf(fid, '%d', 3);
offobj.vertices = reshape(fscanf(fid, '%f', info(1)*3), 3, info(1))';
offobj.faces = reshape(fscanf(fid, '%d', info(2)*4), 4, info(2))';
 % do some translation and rotation
center = (max(offobj.vertices) + min(offobj.vertices)) / 2;
offobj.vertices = bsxfun(@minus, offobj.vertices, center);

if exist('axis', 'var')
    switch axis
        case 'x',
            offobj.vertices(:,1) = offobj.vertices(:,1) * stretch;
        case 'y',
            offobj.vertices(:,2) = offobj.vertices(:,2) * stretch;
        case 'z',
            offobj.vertices(:,3) = offobj.vertices(:,3) * stretch;
        otherwise,
            error('off_loader axis set wrong');
    end
end

%theta = theta * pi / 180;
%R = [cos(theta), -sin(theta), 0;
%     sin(theta), cos(theta) , 0;
%        0      ,    0       , 1];
 %offobj.vertices = offobj.vertices * R;
% These vertices to define faces should be offset by one to follow the matlab convention.

offobj.faces = offobj.faces(:,2:end) + 1; 
fclose(fid);
