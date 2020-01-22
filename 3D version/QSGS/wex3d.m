function arrgrid=wex3d(l,n,cdd,M)

ly = l;  
lx = l;  
lz = l; 
d1 = 0.8; d2 = 0.8; d3 = 0.8; d4 = 0.8; d5 = 0.8; d6 = 0.8; 
d7 = 0.1; d8 = 0.1; d9 = 0.1; d10 = 0.1; d11 = 0.1; d12 = 0.1; d13 = 0.1; d14 = 0.1; d15 = 0.1; d16 = 0.1; d17 = 0.1; d18 = 0.1; 
%n = 0.7; 
%cdd = 0.01;                                                                 
numtotal_need= n * lx * ly *lz;                                                
numsoild = 0;  
arrgrid=M;                                                          
soild = zeros;    
 
 
% ???????????? 
for i=1: lx  
	for j = 1:ly 
        for k = 1:lz 
            if rand( ) < cdd 
                if arrgrid(i,j)~=1
                    numsoild= numsoild+ 1; 
                    arrgrid(i,j,k) = 1/2; 
                    soild(numsoild, 1) = i; 
                    soild(numsoild, 2) = j; 
                    soild(numsoild, 3) = k; 
                end
            end 
        end  
	end  
end  
tnumsoild= numsoild; 
 
 
while tnumsoild< numtotal_need                                                                 
   for index_soild= 1: tnumsoild 
    index_i= soild(index_soild, 1); 
    index_j= soild(index_soild, 2); 
    index_k= soild(index_soild, 3); 
     
	%?? 1 ???????? 
     if index_i< lx 
       i= index_i+1; j = index_j; k= index_k; 
        if (arrgrid(i,j,k)== 0 && rand( ) < d1) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 2 ???????? 
     if index_j< ly 
       i= index_i; j = index_j + 1; k= index_k; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d2) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
 
     %?? 3 ???????? 
     if index_i> 1 
       i= index_i-1; j = index_j; k= index_k; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d3) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end  
 
     %?? 4 ???????? 
     if index_j> 1 
       i= index_i; j = index_j-1; k= index_k; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d4) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end  
 
	%?? 5 ???????? 
     if index_k< lz 
       i= index_i; j = index_j; k= index_k+1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d5) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end  
      
	%?? 6 ???????? 
     if index_k> 1 
       i= index_i; j = index_j; k= index_k-1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d6) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 7 ???????? 
     if index_i< lx && index_j< ly 
       i= index_i+1; j = index_j+1; k= index_k; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d7) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end  
      
	%?? 8 ???????? 
     if index_i> 1 && index_j< ly 
       i= index_i-1; j = index_j+1; k= index_k; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d8) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
           
	%?? 9 ???????? 
     if index_i> 1 && index_j> 1 
       i= index_i-1; j = index_j-1; k= index_k; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d9) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 10 ???????? 
     if index_i< lx && index_j> 1 
       i= index_i+1; j = index_j-1; k= index_k; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d10) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 11 ???????? 
     if index_i< lx && index_k< lz 
       i= index_i+1; j = index_j; k= index_k+1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d11) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 12 ???????? 
     if index_i< lx && index_k> 1 
       i= index_i+1; j = index_j; k= index_k-1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d12) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 13 ???????? 
     if index_i> 1 && index_k> 1 
       i= index_i-1; j = index_j; k= index_k-1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d13) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 14 ???????? 
     if index_i> 1 && index_k< lz 
       i= index_i-1; j = index_j; k= index_k+1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d14) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 15 ???????? 
     if index_j< ly && index_k< lz 
       i= index_i; j = index_j+1; k= index_k+1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d15) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 16 ???????? 
     if index_j< ly && index_k> 1 
       i= index_i; j = index_j+1; k= index_k-1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d16) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 17 ???????? 
     if index_j> 1 && index_k> 1 
       i= index_i; j = index_j-1; k= index_k-1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d17) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
      
	%?? 18 ???????? 
     if index_j> 1 && index_k< lz 
       i= index_i; j = index_j-1; k= index_k+1; 
       if (arrgrid(i,j,k)== 0 && rand( ) < d18) 
          numsoild= numsoild+ 1; arrgrid(i,j,k) = 1/2; 
          soild( numsoild, 1) = i; soild(numsoild, 2) = j; soild(numsoild, 3) = k; 
       end  
     end 
   end 
 tnumsoild=numsoild; 
end  


%{
[x,y,z] = meshgrid(0:.5:10,0:.5:10,0:.5:10);
x=soild(:, 1); 
y=soild(:, 2); 
z=soild(:, 3); 
c=[0.5 0.5 0.5];
scatter3(x,y,z,20,c,'filled','s') 
shading interp
%}