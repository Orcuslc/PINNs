clear
U = [];
for i = 1:27
    load(strcat('solutions/', num2str(i), '.mat'));
    U = [U u(:, 8:8:end)];
end

[UU,S,V] = svd(U);
sig = diag(S);
figure(1)
semilogy(sig)

%figure(2)
x = linspace(-1,1,201);

figure(2)    
subplot(4,2,1); plot(x,UU(:,1)); ylim([-0.1, 0.1]);
subplot(4,2,2); plot(x,-UU(:,2)); ylim([-0.2, 0.2]);
subplot(4,2,3); plot(x,UU(:,3)); ylim([-0.5, 0.5]);
subplot(4,2,4); plot(x,-UU(:,4)); ylim([-0.5, 0.5]);
subplot(4,2,5); plot(x,-UU(:,5)); ylim([-0.5, 0.5]);
subplot(4,2,6); plot(x,UU(:,6)); ylim([-0.5, 0.5]);
subplot(4,2,7); plot(x,-UU(:,7)); ylim([-0.5, 0.5]);

%  critieria 1
s_sig = sum(sig);
cumsum(sig(1:8))/s_sig

%  ctiteria 2
s_sig = sum(sig);
cumsum(sig(1:7).^2)/sum(sig.^2)