startDir = '/Users/shinaushin/Documents/HCP_S1200/DFS'; %specify directory

% Get list of all subfolders.
allSubFolders = genpath(startDir);

% Parse into a cell array.
remain = allSubFolders;

% Initialize variable that will contain the list of filenames so that we can concatenate to it.
listOfFolderNames = {};
while true
	[singleSubFolder, remain] = strtok(remain, ';');
	if isempty(singleSubFolder)
		break;
	end
	listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames);

% Process all dfs files in those folders.
thisFolder = listOfFolderNames{1};
thisFolder = thisFolder(1:length(thisFolder)-1);
fprintf('Processing folder %s\n', thisFolder)

% Get ALL dfs files.

%inner skull
filePattern_isk = sprintf('%s/*.inner_skull.dfs', thisFolder);
baseFileNames_isk = dir(filePattern_isk);
numberOfInnerSkullFiles=length(baseFileNames_isk);

%scalp
filePattern_scalp = sprintf('%s/*.scalp.dfs', thisFolder);
baseFileNames_scalp = dir(filePattern_scalp);
numberOfScalpFiles=length(baseFileNames_scalp);

%brain
filePattern_brain = sprintf('%s/*.brain.dfs', thisFolder);
baseFileNames_brain = dir(filePattern_brain);
numberOfBrainFiles=length(baseFileNames_brain);

if numberOfInnerSkullFiles >=1
    for i=1:numberOfInnerSkullFiles
        %STATEMENT
        fullFileName_isk = fullfile(thisFolder, baseFileNames_isk(i).name);

        isk = readdfs(fullFileName_isk);

        fullFileName_scalp = fullfile(thisFolder, baseFileNames_scalp(i).name);

        scalp = readdfs(fullFileName_scalp);
        
        fullFileName_brain = fullfile(thisFolder, baseFileNames_brain(i).name);

        brain = readdfs(fullFileName_brain);

        %surf2mesh(v,f,p0,p1,keepratio,maxvol,regions,holes,forcebox)
        %inner skull
        tic;
         fprintf('\nSTEP 1:\n Inner Skull \n Volume meshing has started...\n\n');
         [n_isk,e_isk,f_isk]=surf2mesh(isk.vertices,isk.faces,min(isk.vertices(:,1:3))-1,max(isk.vertices(:,1:3))+1,1,100);
         fprintf('\nSTEP 1:\n Meshed volume of Inner Skull has been created!\n\n');
        toc_s2m = toc;

        %outer skull
%         tic;
%          fprintf('\nSTEP 2:\n Outer Skull \n Volume meshing has started...\n\n');
%          [n_osk,e_osk,f_osk]=surf2mesh(osk.vertices,osk.faces,min(osk.vertices(:,1:3))-1,max(osk.vertices(:,1:3))+1,1,100);
%          fprintf('\nSTEP 2:\n Meshed volume of Outer Skull has been created!\n\n');
%         toc_s2m = toc;
        
        n_osk = n_isk - mean(n_isk);
        n_osk = n_osk * 1.025 + mean(n_isk);

        %merge mesh
        [n_skull,e_skull] = mergemesh(n_isk,e_isk,n_osk,e_isk);

        %scalp
        tic;
         fprintf('\nSTEP 3:\n Scalp \n Volume meshing has started...\n\n');
         [n_scalp,e_scalp,f_scalp]=surf2mesh(scalp.vertices,scalp.faces,min(scalp.vertices(:,1:3))-1,max(scalp.vertices(:,1:3))+1,1,100);
         fprintf('\nSTEP 3:\n Meshed volume of Scalp has been created!\n\n');
        toc_s2m = toc;
        
        [n_scalp,e_scalp] = mergemesh(n_scalp,e_scalp);
        
        %brain
        tic;
         fprintf('\nSTEP 4:\n Brain \n Volume meshing has started...\n\n');
         [n_brain,e_brain,f_brain]=surf2mesh(brain.vertices,brain.faces,min(brain.vertices(:,1:3))-1,max(brain.vertices(:,1:3))+1,1,100);
         fprintf('\nSTEP 4:\n Meshed volume of Brain has been created!\n\n');
        toc_s2m = toc;
        
        [n_brain,e_brain] = mergemesh(n_brain,e_brain);
        
        % plot mesh
%         figure;
%         plotmesh(n_skull,e_skull);

        %save stl - savestl(node,elem,fname,solidname)
        ix=strfind(baseFileNames_isk(i).name,'.');  % get the underscore locations
        id=baseFileNames_isk(i).name(1:ix(1)-1);     % return the substring up to 2nd underscore

        s1='skull_';
        ext='.stl';
        fileName= strcat("./STL/skull/",s1,id,ext);
        savestl(n_skull,e_skull,fileName);
        
        s1='scalp_';
        ext='.stl';
        fileName= strcat("./STL/scalp/",s1,id,ext);
        savestl(n_scalp,e_scalp,fileName);
        
        s1='brain_';
        ext='.stl';
        fileName= strcat("./STL/brain/",s1,id,ext);
        savestl(n_brain,e_brain,fileName);
    end
else 
    fprintf('     Folder %s does not have correct number of files.\n', thisFolder);
end 

disp(id);
disp("Finished.");
