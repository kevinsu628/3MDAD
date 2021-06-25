files = dir('./mat/*.mat');
for file = files'
    matFile = fullfile("./mat", file.name);
    field = file.name(1:end-4);
    csvFile = fullfile("./csv", field+".csv");
    data = load(matFile).(field);
    writetable(data, csvFile);
end