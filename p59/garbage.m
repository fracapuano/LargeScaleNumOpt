figure(2)

hold on
plot(grads_corr, "DisplayName", "A")
plot(gradients_INN, "DisplayName", "B")
legend show
hold off

set(gca, "XScale", "log", "YScale", "log")