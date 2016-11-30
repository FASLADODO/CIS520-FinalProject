%% Script to load relevant GloVe data from txt file

numComponents = 25; % Options: 25, 50, 100, 200

% read in topwords.csv first
fid = fopen('topwords.csv');
C = textscan(fid, '%s');
fclose(fid);

% initialize data structures to store data
wordmap = containers.Map(C{1}, 1:10005);
vecs = zeros(10000, numComponents);
happy = zeros(1, numComponents);
sad = zeros(1, numComponents);

fid = fopen(['glove.twitter.27B/glove.twitter.27B.' int2str(numComponents) 'd.txt']);

tline = fgetl(fid);
while ischar(tline)
    line_data = textscan(tline, ['%s ', repmat('%f ', 1, numComponents)]);
    word = line_data{1}{1};
    if wordmap.isKey(word)
        vecs(wordmap(word), :) = cell2mat(line_data(2:(numComponents+1)));
    elseif isequal(word(1), '#')
       % Try stripping the hashtag
       word = word(2:end);
       if wordmap.isKey(word)
        vecs(wordmap(word), :) = cell2mat(line_data(2:(numComponents+1)));
       end
    elseif isequal(word(1), 'happy')
        happy = cell2mat(line_data(2:(numComponents+1)));
    elseif isequal(word(1), 'sad')
        sad = cell2mat(line_data(2:(numComponents+1)));
    end
    tline = fgetl(fid);
end
fclose(fid);