%% Script to load relevant GloVe data from txt file

numComponents = 25; % Options: 25, 50, 100, 200

% read in topwords.csv first
fid = fopen('topwords.csv');
C = textscan(fid, '%s');
fclose(fid);

% initialize data structures to store data
wordmap = containers.Map(C{1}, 1:10005);
words = zeros(10005, 1);
vecs = zeros(10005, numComponents);

fid = fopen(['glove.twitter.27B/glove.twitter.27B.' int2str(numComponents) 'd.txt']);

tline = fgetl(fid);
counter = 1;
while ischar(tline)
    line_data = textscan(tline, ['%s ', repmat('%f ', 1, numComponents)]);
    word = line_data{1}{1};
    if wordmap.isKey(word)
        words(counter) = wordmap(word);
        vecs(counter, :) = cell2mat(line_data(2:(numComponents+1)));
        counter = counter + 1;
    elseif isequal(word(1), '#')
       % Try stripping the hashtag
       word = word(2:end);
       if wordmap.isKey(word)
        words(counter) = wordmap(word);
        vecs(counter, :) = cell2mat(line_data(2:(numComponents+1)));
        counter = counter + 1;
       end
    end
    tline = fgetl(fid);
end
fclose(fid);