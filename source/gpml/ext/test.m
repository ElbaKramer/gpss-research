%% original gp_delta test
clear all; close all;
x = 0:0.1:10;
x = x';
y = randn(numel(x),1)*5 + 10;
inf = @infDelta;
mean = @meanZero;
cov = {@covSum, {@covConst, @covNoise}};
lik = @likDelta;
hyp.mean = [];
hyp.cov = zeros(2,1);
hyp.lik = [];
opt = minimize(hyp, 'gp_delta', -500, inf, mean, cov, lik, x, y);
fprintf('gp_delta passed\n\n');
%% gp_rel_v1 test
clear all; close all;
load('/home/yunseong/Git/gpss-research/examples/data/house4.mat');
inf = @infDelta;
mean = @meanZero;
cov = {@covSum, {@covSEiso, @covNoise}};
lik = @likDelta;
hyp.mean = [];
hyp.cov = zeros(3,1);
hyp.lik = [];
nll = gp_rel_v1(hyp, inf, mean, cov, lik, X, y);
opt = minimize(hyp, 'gp_rel_v1', -500, inf, mean, cov, lik, X, y);
nll_opt = gp_rel_v1(opt, inf, mean, cov, lik, X, y);
fprintf('gp_rel_v1 passed\n\n');
%% gp_rel_v2 test
clear all; close all;
load('/home/yunseong/Git/gpss-research/examples/data/house4.mat');
inf = @infDelta;
mean = @meanZero;
cov = {@covSum, {@covSEiso, @covNoise}};
lik = @likDelta;
hyp.mean = [];
hyp.cov = zeros(3,1);
hyp.lik = [];
hyp.norm = zeros(4*2,1);
nll = gp_rel_v2(hyp, inf, mean, cov, lik, X, y);
opt = minimize(hyp, 'gp_rel_v2', -500, inf, mean, cov, lik, X, y);
nll_opt = gp_rel_v2(opt, inf, mean, cov, lik, X, y);
fprintf('gp_rel_v2 passed\n\n');