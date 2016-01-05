clear all; close all;
%%
x = 0:0.1:10;
x = x';
y1 = randn(numel(x),1) + x;
y2 = randn(numel(x),1) + 10*x + 10;

X = {x,x};
y = {y1,y2};

inf = @infDelta;
mean = @meanZero;
cov = {@covSum, {@covLIN, @covNoise}};
lik = @likDelta;

hyp.mean = [];
hyp.cov = zeros(1,1);
hyp.lik = [];

nll1 = gp_rel_v1(hyp, inf, mean, cov, lik, X, y);

hyp.norm = log([1;0.1;10;10]);
nll2 = gp_rel_v2(hyp, inf, mean, cov, lik, X, y);

%%
opt = minimize(hyp, 'gp_rel_v2', -500, inf, mean, cov, lik, X, y);