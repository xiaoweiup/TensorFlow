function [r_square,rs] = rsqrgen(ra,output)

size_ra=size(ra);

for i=1:1:size_ra(1,1)
    tmp_simu(i)=ra(i)-mean(ra);
    tmp_pKd(i)=output(i)-mean(output);
    fenzi(i)=tmp_simu(i)*tmp_pKd(i);
    fenmu1(i)=tmp_simu(i)*tmp_simu(i);
    fenmu2(i)=tmp_pKd(i)*tmp_pKd(i);
end


rs=sum(fenzi)/(sqrt(sum(fenmu1))*sqrt(sum(fenmu2)));

r_square=rs^2;
