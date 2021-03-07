function [mean_dice_coeff, errci] = mean_errci(a_dice_coeff)

mean_dice_coeff = mean(bootstrp(1000, @mean, a_dice_coeff));
ci_dice_coeff = bootci(1000, @mean, a_dice_coeff);

errorhigh = ci_dice_coeff(2);
errorlow = ci_dice_coeff(1);
errhigh = errorhigh - mean_dice_coeff;
errlow = mean_dice_coeff - errorlow;

errci(:, :, 1) = errlow;
errci(:, :, 2) = errhigh;

end