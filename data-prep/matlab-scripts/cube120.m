%masking noise injection for skull models with voxel resolution of 120x120x120
types = ["scalp"; "skull"; "brain"];

data_path='/Users/shinaushin/Documents/MATLAB/JHU/Spring 2020/Research/skull-complete/data-prep/matlab-scripts/120';

full_data_path = strcat(data_path, '/', types(1), '_MATs'); % '/' num2str(data_size)]; 

phases = {'test'}; %noise injection is only applied on the testing set

mat_list =[full_data_path];

mat_files=dir(fullfile(mat_list, '*.mat'));

for i = 1:length(mat_files)
    for j=1:2
        full_data_path = strcat(data_path, '/', types(j), '_MATs'); % '/' num2str(data_size)]; 

        mat_list =[full_data_path];

        mat_files=dir(fullfile(mat_list, '*.mat'));

        %assuring only .mat files are processed
        if strcmp(mat_files(i).name, '.') || strcmp(mat_files(i).name, '..') || mat_files(i).isdir == 1 || ~strcmp(mat_files(i).name(end-2:end), 'mat')
            continue;
        end

        filename = strcat(mat_list, '/', mat_files(i).name);

        load(filename, 'instance');
        defected=instance(:,:,:);

        enough=0;
        while enough == 0
            %selecting the dimensions of the cube. possible dimensions: 20x20x20, 24x24x24, 28x28x28 and 32x32x32
            if j == 1
                %choosing the coordinates in the model where to remove the voxel cube [bb,cc,dd]
                bb = randperm(120,1);
                cc = randperm(120,1);
                dd = randi([68 120],1,1);
                l1 = randi([12 18],1,1);
                l2 = randi([12 18],1,1);
            else
                l1 = l1 - 2;
                l2 = l2 - 2;
            end
            
            d = norm([bb-60,cc-60,dd-50]);
            l1_norm = l1 * d / 50;
            l2_norm = l2 * d / 50;
            
            a = l1_norm / d;
            b = l2_norm / d;
            A = 1/a;
            B = 1/b;
            rotmat = vrrotvec2mat(vrrotvec([bb-60,cc-60,dd-50], [0,0,1]));
            idx_remove = [];
            for z=50:120
                for y=1:120
                    for x=1:120
                        x_prime = [x-60;y-60;z-50];
                        x_prime = rotmat*x_prime;
                        if x_prime(3) - sqrt(A^2*x_prime(1)^2+B^2*x_prime(2)^2) > 0 && instance(x,y,z) == 1
                            % defected(x,y,z) = 0;
                            idx_remove = [idx_remove; [x, y, z]];
                        end
                    end
                end
            end
            
            % soma=sum(sum(sum(instance(b,c,d) == 1))); %count the number of voxels that will be cut away from the skull model by the cube
            % soma = sum(sum(sum(instance(idx_remove) == 1)));
            
            % fprintf('2. soma %d \n', soma);
            % fprintf('3. filename %s \n', filename);
            if j == 1
                if size(idx_remove,1) < 600 %if the number of voxels to be cut is less than 400, generate another cube until the condition of more than 400 voxels to cut is met
                    %fprintf('4. hey %d \n', enough);
                    continue;
                else
                    for k=1:size(idx_remove,1)
                        defected(idx_remove(k,1), idx_remove(k,2), idx_remove(k,3)) = 0;
                    end
                end
            else
                for k=1:size(idx_remove,1)
                    defected(idx_remove(k,1), idx_remove(k,2), idx_remove(k,3)) = 0;
                end
            end
            enough=1;
        end

        % defected(b,c,d) = 0; %cut the cube from the skull model
        % defected(idx_remove) = 0;
        % plot_occupancy(defected);
        % disp([bb, cc, dd]);
        % plot_cross_sections(defected, bb, cc, dd);
        
        save(filename, 'instance','defected'); %save the intact skull and the resultant artificially corrupted skull model in the same .mat file but in different variables
    end
    disp(mat_files(i).name);
    % pause;
    % close all;
end

%{
figure;    
p = isosurface(squeeze(new_instance),0.5) ;
 patch( p,'facecolor',[1 0 0],'edgecolor','none'), camlight;view(3)  

axis equal 
axis on      
lighting gouraud
pause;

close all;
%}