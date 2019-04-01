% %% u_high
% %% train
% mu = 1.0:0.25:7.5;
% 
% % for i = 1:length(mu)
% %     [u, x] = FD_main_shock(1.0, 200, mu(i)/(50*pi), i);
% % end
% 
% U = [];
% for i = 1:length(mu)
%     load(strcat('solutions_train/', num2str(i), '.mat'));
%     U = [U u(:, 9:8:end)];
% end
% 
% %% test
% mu = [1.125, 3.125, 5.125, 7.125];
% t = [0.25, 0.5, 0.75, 0.90, 1.00];
% 
% % for i = 1:length(mu)
% %     [u, x] = FD_main_shock(1.0, 200, mu(i)/(50*pi), i);
% % end
% 
% U = [];
% for i = 1:length(mu)
%     load(strcat('solutions/', num2str(i), '.mat'));
%     U = [U u(:, 2000*t+1)];
% end

%% u_low
%% train
% mu = 1.0:0.25:7.5;
% 
% for i = 1:length(mu)
%     [u, x] = FD_main_shock(1.0, 20, mu(i)/(50*pi), i);
% end
% U = [];
% for i = 1:length(mu)
%     load(strcat('solutions/', num2str(i), '.mat'));
%     U = [U u(:, 9:8:end)];
% end

%% test
% mu = [1.125, 3.125, 5.125, 7.125];
% t = [0.25, 0.5, 0.75, 0.90, 1.00];
% 
% for i = 1:length(mu)
%     [u, x] = FD_main_shock(1.0, 20, mu(i)/(50*pi), i);
% end
% 
% U = [];
% for i = 1:length(mu)
%     load(strcat('solutions/', num2str(i), '.mat'));
%     U = [U u(:, 2000*t+1)];
% end

mu = 0.01/pi;
[u, x] = FD_main_shock(1.0, 200, mu, 1);
