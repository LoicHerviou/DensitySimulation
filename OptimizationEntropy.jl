include("GeneralMethods.jl")
include("SuperOperator.jl")


using SymPy
using FileIO
#
# function buildKernels(maxLen, Hcommuting)
#     TrRight=buildTraceMatrixRight(maxLen-1);
#     TrLeft=buildTraceMatrixLeft(maxLen-1);
#     ###For the left link
#     leftCommuter=buildSuperOperatorLeft(kron(speye(2^(maxLen-1)), Hcommuting))
#     MatJoin=sparse(vcat(TrRight, TrLeft, TrLeft*leftCommuter));
#     kernelLeft=nullspace(full(MatJoin))
#     ###For the right link
#     rightCommuter=buildSuperOperatorLeft(kron(Hcommuting, speye(2^(maxLen-1))))
#     MatJoin=sparse(vcat(TrLeft, TrRight, TrRight*rightCommuter));
#     kernelRight=nullspace(full(MatJoin))
#     return kernelLeft, kernelRight
# end

function hermitianed(vect)
    len=getLength(length(vect));
    return vect+view(reshape(vect, (2^len, 2^len))', :)
end

# function makeHermitian(kernel)
#     herm=zeros(Complex128, size(kernel));
#     herm[:, 1]=hermitianed(kernel[:, 1])
#     count=2
#     for j=2:(size(kernel)[2])
#         temp=symmetrize(kernel[:, j])
#         herm[:, count]=copy(temp)
#         if rank(herm)==count
#             count+=1
#         end
#     end
#     println(count)
#     herm[:, count]=hermitianed(1im*kernel[:, 1])
#      count+=1
#      for j=2:(size(kernel)[2])
#          temp=symmetrize(1im*kernel[:, j])
#          herm[:, count]=copy(temp)
#          if rank(herm)==count
#              count+=1
#              if count>size(kernel)[2]
#                  break
#              end
#          end
#      end
#     return herm
# end
#
#
# function generateHermitianKernel(maxLen)
#     sx=[0 1; 1 0]; sz=-[1  0 ; 0 -1]; sy=[0 -1im; 1im 0]
#     kernelLeft, kernelRight=buildKernels(maxLen, sparse(sz))
#     hermitianKernelLeft=makeHermitian(kernelLeft)
#     hermitianKernelRight=makeHermitian(kernelRight)
#     return hermitianKernelLeft, hermitianKernelRight
# end



function makeHermitian(kernel)
    herm=zeros(Complex{Int64}, length(kernel[1]), length(kernel))
    herm[:, 1]=N(hermitianed(kernel[1]))
    count=2
    for j=2:length(kernel)
        println(count)
        herm[:, count]=N(hermitianed(kernel[j]))
        if rank(herm)==count
            count+=1
        end
    end
    herm[:, count]= convert(Array{Complex{Int64},1},  N(hermitianed(1im*kernel[1])))
     count+=1
     for j=2:length(kernel)
         herm[:, count]= convert(Array{Complex{Int64},1},  N(hermitianed(1im*kernel[j])))
         println(count)
         if rank(herm)==count
             count+=1
             if count>length(kernel)
                 break
             end
         end
     end
    return herm
end


function buildKernels(maxLen, Hcommuting)
    TrRight=buildTraceMatrixRight(maxLen-1, typed=SymPy.Sym);
    TrLeft=buildTraceMatrixLeft(maxLen-1, typed=SymPy.Sym);
    ###For the left link
    leftCommuter=buildSuperOperatorLeft(kron(speye(SymPy.Sym, 2^(maxLen-1)), Hcommuting))
    MatJoin=sparse(vcat(TrRight, TrLeft, TrLeft*leftCommuter));
    kernelLeft=sparse.(MatJoin[:nullspace]())
    ###For the right link
    rightCommuter=buildSuperOperatorLeft(kron(Hcommuting, speye(SymPy.Sym,2^(maxLen-1))))
    MatJoin=sparse(vcat(TrLeft, TrRight, TrRight*rightCommuter));
    kernelRight=sparse.(MatJoin[:nullspace]())
    return kernelLeft, kernelRight
end


function generateHermitianKernel(maxLen)
    sz=-Sym[1  0 ; 0 -1];
    kernelLeft, kernelRight=buildKernels(maxLen, sparse(sz))
    hermitianKernelLeft=sparse(makeHermitian(kernelLeft))
    hermitianKernelRight=sparse(makeHermitian(kernelRight))
    return hermitianKernelLeft, hermitianKernelRight
end



function mainGeneration(maxLen)
    hermLeft, hermRight=generateHermitianKernel(maxLen)
    FileIO.save(string("kernelIsing-len-", maxLen, ".jld2"), "len", maxLen, "kernelLeft", hermLeft, "kernelRight", hermRight)
end



function extractLowerEigenvalue(vect)
    len=getLength(length(vect));
    a, _=eigs(reshape(view(vect, :), (2^len, 2^len))-10*I, nev=1, which=:LM)
    return real(a[1])+10.0
end



function improveRho!(rho, kernel; de=0.0001, step=100, dx=0.1)
    gradient=zeros(size(kernel)[2])
    temprho=similar(rho)
    totalmodif=zeros(size(kernel)[2])
    for x=1:step
        target=extractLowerEigenvalue(rho)
        for j=1:length(gradient)
            temprho=rho+de*kernel[:, j]
            gradient[j]=(extractLowerEigenvalue(temprho)-target)/de*dx
        end
        for j=1:length(gradient)
            @. rho=rho+gradient[j]*kernel[:, j]
            @. totalmodif=totalmodif+gradient
        end
    end
end


matrho=rand(Complex128, 16, 16); matrho=matrho+matrho';
rho=reshape(copy(matrho), length(matrho))
target=extractLowerEigenvalue(rho)
 improveRho!(rho, kernelLeft)
finished=extractLowerEigenvalue(rho)
