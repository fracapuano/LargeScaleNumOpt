fterms_lin = @(gradf_norm) 0.5; 
fterms_suplin = @(gradf_norm) min([0.5, sqrt(gradf_norm)]); 
fterms_quad = @(gradf_norm) min([0.5, gradf_norm]); 

save forcing_terms.mat