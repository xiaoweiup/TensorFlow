function RMSD = rmsdgen(ra,output)

size_ra=size(ra);

for i=1:1:size_ra(1,1)
    ss(i)=(ra(i)-output(i))^2;
end

RMSD=sqrt(sum(ss)/size(output,1));





