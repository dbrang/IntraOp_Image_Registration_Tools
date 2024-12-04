% INTRAOP IMAGE REGISTRATION TOOL v1
%
% Description:
% This MATLAB script provides an interactive tool for image registration.
% It allows users to load two images: one with a grid (reference) and one
% without a grid (target). Using the provided interface, users can:
%
% 1. Load and display the grid and no-grid images.
% 2. Select control points to align (register) the no-grid image to the grid image.
% 3. Save the control points, transformed image, and other data for later use.
% 4. Toggle between the grid image, no-grid image, and the transformed image.
%
% The tool dynamically adjusts its layout to fit the figure window size,
% ensuring ease of use on various screen resolutions.
%
% Features:
% - Load images (grid and no-grid).
% - Select control points for registration.
% - Apply a projective transformation to register the no-grid image.
% - Toggle visibility between images.
% - Save all relevant data to a .mat file.
%
% Usage:
% - Run the script and use the buttons to load images, register them, and
%   save the results.
%
% Requirements:
% - MATLAB Image Processing Toolbox for `cpselect` and `imwarp`.
%
% Author:
% David Brang
%
% Copyright (c) 2024 David Brang
% All rights reserved.
%
% License: CC BY-NC 4.0
%
% Version: 1.0
% Date: November 21, 2024
%
% Contact:
% For questions or bug reports, contact David Brang at djbrang@umich.edu

function IntraOp_Image_Registration_Manual()
% Create the main figure
fig = figure('Name', 'Image Registration Tool', 'NumberTitle', 'off', ...
    'WindowState', 'maximized');

% Initialize UserData with placeholders for control points, images, and buttons
fig.UserData.cp1 = [];
fig.UserData.cp2 = [];
fig.UserData.transformed_img1 = []; % Transformed no-grid image
fig.UserData.img1 = []; % No-grid image
fig.UserData.img2 = []; % Grid image
fig.UserData.hNoGrid = []; % Handle for no-grid image
fig.UserData.hGrid = []; % Handle for grid image
fig.UserData.hNoGridTransformed = []; % Handle for transformed no-grid image
fig.UserData.buttonHandles = struct();

% Create buttons and store their handles
fig.UserData.buttonHandles.loadMat = uicontrol('Style', 'pushbutton', 'String', 'Load Previous .mat', ...
    'Callback', @(~, ~) loadMat(fig));

fig.UserData.buttonHandles.loadNoGrid = uicontrol('Style', 'pushbutton', 'String', 'Load No Grid Image', ...
    'Callback', @(~, ~) loadNoGrid(fig));

fig.UserData.buttonHandles.loadGrid = uicontrol('Style', 'pushbutton', 'String', 'Load Grid Image', ...
    'Callback', @(~, ~) loadGrid(fig));

fig.UserData.buttonHandles.toggleTransform = uicontrol('Style', 'togglebutton', 'String', 'Toggle Image', ...
    'Callback', @(src, ~) toggleImages(fig, src));

fig.UserData.buttonHandles.selectControlPoints = uicontrol('Style', 'pushbutton', 'String', 'Select Control Points', ...
    'Callback', @(~, ~) selectControlPoints(fig));

fig.UserData.buttonHandles.saveData = uicontrol('Style', 'pushbutton', 'String', 'Save Data', ...
    'Callback', @(~, ~) saveData(fig));

fig.UserData.buttonHandles.quit = uicontrol('Style', 'pushbutton', 'String', 'Quit', ...
    'Callback', @(~, ~) delete(fig));

% Set the resize function after initialization
fig.SizeChangedFcn = @(src, ~) resizeButtons(src);

% Display placeholder message in axes
hAx = axes(fig, 'Position', [0, 0, 1, 1]); % Fill entire figure
disableDefaultInteractivity(hAx); % Disable all default interactions
axis(hAx, 'off'); % Turn off axis ticks and labels
text(0.5, 0.5, 'Load images or previous .mat to begin.', 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', 36, 'Parent', hAx);

% Callback to resize buttons dynamically
    function resizeButtons(src)
        % Retrieve button handles from UserData
        buttonHandles = src.UserData.buttonHandles;

        % Get updated figure dimensions
        figWidth = src.Position(3);
        figHeight = src.Position(4);

        % Define button dimensions and spacing
        buttonWidth = figWidth * 0.1; % 10% of figure width
        buttonHeight = figHeight * 0.05; % 5% of figure height
        buttonSpacing = figWidth * 0.015; % 1.5% of figure width
        quitButtonWidth = buttonWidth / 2; % Quit button is half the size

        % Update button positions
        set(buttonHandles.loadMat, 'Position', [buttonSpacing, 10, buttonWidth, buttonHeight]);
        set(buttonHandles.loadNoGrid, 'Position', [buttonSpacing * 2 + buttonWidth, 10, buttonWidth, buttonHeight]);
        set(buttonHandles.loadGrid, 'Position', [buttonSpacing * 3 + buttonWidth * 2, 10, buttonWidth, buttonHeight]);
        set(buttonHandles.toggleTransform, 'Position', [buttonSpacing * 4 + buttonWidth * 3, 10, buttonWidth, buttonHeight]);
        set(buttonHandles.selectControlPoints, 'Position', [buttonSpacing * 5 + buttonWidth * 4, 10, buttonWidth, buttonHeight]);
        set(buttonHandles.saveData, 'Position', [buttonSpacing * 6 + buttonWidth * 5, 10, buttonWidth, buttonHeight]);
        set(buttonHandles.quit, 'Position', [figWidth - quitButtonWidth - buttonSpacing, 10, quitButtonWidth, buttonHeight]);
    end

% Function to load no-grid image
    function loadNoGrid(fig)
        [file, path] = uigetfile('*.jpg;*.png;*.tif', 'Select No Grid Image');
        if isequal(file, 0)
            return;
        end
        img1 = imread(fullfile(path, file));
        fig.UserData.img1 = img1;

        % Ensure the axes can handle multiple images
        axes(hAx);
        hold(hAx, 'on');

        % Remove placeholder text if present
        delete(findall(fig, 'Type', 'text', 'Parent', hAx));
        axis(hAx, 'on'); % Enable axes functionality for images

        % Display image
        if isempty(fig.UserData.hNoGrid) || ~isvalid(fig.UserData.hNoGrid)
            % fig.UserData.hNoGrid = imshow(img1, 'Parent', hAx);
            fig.UserData.hNoGrid = imshow(img1, 'Parent', hAx, 'InitialMagnification', 'fit');
        else
            set(fig.UserData.hNoGrid, 'CData', img1, 'Visible', 'on');
        end
        % Hide other images if they exist
        if ~isempty(fig.UserData.hGrid) && isvalid(fig.UserData.hGrid)
            set(fig.UserData.hGrid, 'Visible', 'off');
        end
        if ~isempty(fig.UserData.hNoGridTransformed) && isvalid(fig.UserData.hNoGridTransformed)
            set(fig.UserData.hNoGridTransformed, 'Visible', 'off');
        end
    end

% Function to load grid image
    function loadGrid(fig)
        [file, path] = uigetfile('*.jpg;*.png;*.tif', 'Select Grid Image');
        if isequal(file, 0)
            return;
        end
        img2 = imread(fullfile(path, file));
        fig.UserData.img2 = img2;

        % Ensure the axes can handle multiple images
        axes(hAx);
        hold(hAx, 'on');

        % Remove placeholder text if present
        delete(findall(fig, 'Type', 'text', 'Parent', hAx));
        axis(hAx, 'on'); % Enable axes functionality for images

        % Display image
        if isempty(fig.UserData.hGrid) || ~isvalid(fig.UserData.hGrid)
            %fig.UserData.hGrid = imshow(img2, 'Parent', hAx);
            fig.UserData.hGrid = imshow(img2, 'Parent', hAx, 'InitialMagnification', 'fit');
        else
            set(fig.UserData.hGrid, 'CData', img2, 'Visible', 'on');
        end
        % Hide other images if they exist
        if ~isempty(fig.UserData.hNoGrid) && isvalid(fig.UserData.hNoGrid)
            set(fig.UserData.hNoGrid, 'Visible', 'off');
        end
        if ~isempty(fig.UserData.hNoGridTransformed) && isvalid(fig.UserData.hNoGridTransformed)
            set(fig.UserData.hNoGridTransformed, 'Visible', 'off');
        end
    end

% Function to toggle images
    function toggleImages(fig, src)
        % Retrieve handles
        hNoGrid = fig.UserData.hNoGrid;
        hGrid = fig.UserData.hGrid;
        hNoGridTransformed = fig.UserData.hNoGridTransformed;

        % Check if the transformed image exists and is valid
        if ~isempty(hNoGridTransformed) && isvalid(hNoGridTransformed)
            if src.Value == 1
                % Show transformed image, hide others
                set(hNoGridTransformed, 'Visible', 'on');
                if ~isempty(hNoGrid) && isvalid(hNoGrid)
                    set(hNoGrid, 'Visible', 'off');
                end
                if ~isempty(hGrid) && isvalid(hGrid)
                    set(hGrid, 'Visible', 'off');
                end
            else
                % Show original images, hide transformed
                set(hNoGridTransformed, 'Visible', 'off');
                if ~isempty(hNoGrid) && isvalid(hNoGrid)
                    set(hNoGrid, 'Visible', 'on');
                end
                if ~isempty(hGrid) && isvalid(hGrid)
                    set(hGrid, 'Visible', 'on');
                end
            end
        else
            % If transformed image doesn't exist, toggle between original images
            if src.Value == 1
                if ~isempty(hGrid) && isvalid(hGrid)
                    set(hGrid, 'Visible', 'off');
                end
                if ~isempty(hNoGrid) && isvalid(hNoGrid)
                    set(hNoGrid, 'Visible', 'on');
                end
            else
                if ~isempty(hGrid) && isvalid(hGrid)
                    set(hGrid, 'Visible', 'on');
                end
                if ~isempty(hNoGrid) && isvalid(hNoGrid)
                    set(hNoGrid, 'Visible', 'off');
                end
            end
        end
    end

% Function to select control points and register no-grid image to grid image
    function selectControlPoints(fig)
        if isempty(fig.UserData.img1) || isempty(fig.UserData.img2)
            msgbox('Please load both images before selecting control points.', 'Error', 'error');
            return;
        end

        img1 = fig.UserData.img1; % No-grid image
        img2 = fig.UserData.img2; % Grid image
        cp1 = fig.UserData.cp1;
        cp2 = fig.UserData.cp2;

        if isempty(cp1) || isempty(cp2)
            [cp1, cp2] = cpselect(img1, img2, 'Wait', true);
        else
            [cp1, cp2] = cpselect(img1, img2, cp1, cp2, 'Wait', true);
        end

        if size(cp1, 1) >= 4 && size(cp2, 1) == size(cp1, 1)
            tform = fitgeotform2d(cp1, cp2, 'projective');
            outputView = imref2d(size(img2));
            transformed_img1 = imwarp(img1, tform, 'OutputView', outputView);

            fig.UserData.cp1 = cp1;
            fig.UserData.cp2 = cp2;
            fig.UserData.transformed_img1 = transformed_img1;

            axes(hAx);
            hold(hAx, 'on');
            if isempty(fig.UserData.hNoGridTransformed) || ~isvalid(fig.UserData.hNoGridTransformed)
                fig.UserData.hNoGridTransformed = imshow(transformed_img1, 'Parent', hAx);
            else
                set(fig.UserData.hNoGridTransformed, 'CData', transformed_img1, 'Visible', 'on');
            end

            if ~isempty(fig.UserData.hNoGrid) && isvalid(fig.UserData.hNoGrid)
                set(fig.UserData.hNoGrid, 'Visible', 'off');
            end
            if ~isempty(fig.UserData.hGrid) && isvalid(fig.UserData.hGrid)
                set(fig.UserData.hGrid, 'Visible', 'off');
            end

            %msgbox('No-grid image registered to grid image successfully.', 'Success');
        else
            msgbox('Insufficient control points selected. At least 4 pairs are required.', 'Error', 'error');
        end
    end

% Function to save data
function saveData(fig)
    if isempty(fig.UserData.img1) || isempty(fig.UserData.img2)
        msgbox('No images to save. Please load images first.', 'Error', 'error');
        return;
    end
    
    [file, path] = uiputfile('*.mat', 'Save Data As', 'registration_data.mat');
    if isequal(file, 0)
        return;
    end

    % Save the .mat data
    img1 = fig.UserData.img1;
    img2 = fig.UserData.img2;
    cp1 = fig.UserData.cp1;
    cp2 = fig.UserData.cp2;
    transformed_img1 = fig.UserData.transformed_img1;
    save(fullfile(path, file), 'img1', 'img2', 'cp1', 'cp2', 'transformed_img1');

    % Save the images
    imwrite(fig.UserData.hGrid.CData, fullfile(path, 'Grid.jpg'));
    imwrite(fig.UserData.hNoGrid.CData, fullfile(path, 'NoGrid.jpg'));
    if ~isempty(fig.UserData.hNoGridTransformed)
        imwrite(fig.UserData.hNoGridTransformed.CData, fullfile(path, 'RegisteredNoGrid.jpg'));
    end

    % Create a GIF transitioning between the grid and registered no-grid images
    if ~isempty(fig.UserData.hGrid) && ~isempty(fig.UserData.hNoGridTransformed)
        gridImg = fig.UserData.hGrid.CData;
        transformedImg = fig.UserData.hNoGridTransformed.CData;
        
        % Ensure both images are the same size
        if ~isequal(size(gridImg), size(transformedImg))
            transformedImg = imresize(transformedImg, [size(gridImg, 1), size(gridImg, 2)]);
        end
        
        % Create the animation
        gifFile = fullfile(path, 'Registered_Animation.gif');
        for alpha = linspace(0, 1, 10) % Linearly change alpha in 10 steps
            blendedImg = uint8((1 - alpha) * double(gridImg) + alpha * double(transformedImg));
            [imind, cm] = rgb2ind(blendedImg, 256); % Convert to indexed image
            if alpha == 0
                imwrite(imind, cm, gifFile, 'gif', 'LoopCount', inf, 'DelayTime', 0.1);
            else
                imwrite(imind, cm, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
            end
        end
    end

    msgbox('Data and images saved successfully.', 'Success');
end

% Function to load previous .mat file
    function loadMat(fig)
        [file, path] = uigetfile('*.mat', 'Select .mat File');
        if isequal(file, 0)
            return;
        end
        data = load(fullfile(path, file), 'img1', 'img2', 'cp1', 'cp2', 'transformed_img1');

        % Update UserData with loaded data
        if isfield(data, 'img1')
            fig.UserData.img1 = data.img1;
        end
        if isfield(data, 'img2')
            fig.UserData.img2 = data.img2;
        end
        if isfield(data, 'cp1')
            fig.UserData.cp1 = data.cp1;
        end
        if isfield(data, 'cp2')
            fig.UserData.cp2 = data.cp2;
        end
        if isfield(data, 'transformed_img1')
            fig.UserData.transformed_img1 = data.transformed_img1;
        end

        % Display loaded images
        axes(fig.CurrentAxes); % Ensure the correct axes is active
        hold(fig.CurrentAxes, 'on');
        if ~isempty(fig.UserData.img1)
            if isempty(fig.UserData.hNoGrid) || ~isvalid(fig.UserData.hNoGrid)
                fig.UserData.hNoGrid = imshow(fig.UserData.img1, 'Parent', fig.CurrentAxes);
            else
                set(fig.UserData.hNoGrid, 'CData', fig.UserData.img1, 'Visible', 'on');
            end
        end
        if ~isempty(fig.UserData.img2)
            if isempty(fig.UserData.hGrid) || ~isvalid(fig.UserData.hGrid)
                fig.UserData.hGrid = imshow(fig.UserData.img2, 'Parent', fig.CurrentAxes);
            else
                set(fig.UserData.hGrid, 'CData', fig.UserData.img2, 'Visible', 'on');
            end
        end
        if ~isempty(fig.UserData.transformed_img1)
            if isempty(fig.UserData.hNoGridTransformed) || ~isvalid(fig.UserData.hNoGridTransformed)
                fig.UserData.hNoGridTransformed = imshow(fig.UserData.transformed_img1, 'Parent', fig.CurrentAxes);
                set(fig.UserData.hNoGridTransformed, 'Visible', 'off');
            else
                set(fig.UserData.hNoGridTransformed, 'CData', fig.UserData.transformed_img1, 'Visible', 'off');
            end
        end
        % Set toggle button state and call toggleImages
        toggleButton = fig.UserData.buttonHandles.toggleTransform;
        set(toggleButton, 'Value', 1); % Set toggle button state to ON
        toggleImages(fig, toggleButton); % Call toggleImages with the updated state

        % Remove placeholder text if present
        delete(findall(fig, 'Type', 'text', 'Parent', hAx));
        %axis(hAx, 'on'); % Enable axes functionality for images

        %msgbox('Data loaded successfully.', 'Success');
    end
end
