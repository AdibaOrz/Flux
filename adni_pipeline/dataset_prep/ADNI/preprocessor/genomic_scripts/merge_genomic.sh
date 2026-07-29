# Get the multi allelic SNPs
echo "./liftovered/ADNI_2_1" > ./liftovered/all_datasets.txt
echo "./liftovered/ADNI_2_2" >> ./liftovered/all_datasets.txt
echo "./liftovered/ADNI_3_1" >> ./liftovered/all_datasets.txt
echo "./liftovered/ADNI_3_2" >> ./liftovered/all_datasets.txt
plink --bfile ./liftovered/ADNI_1 --merge-list ./liftovered/all_datasets.txt --make-bed --out ./liftovered/ADNI_merged


# Deal with multi allelic SNPs
plink --bfile ./liftovered/ADNI_1 --exclude ./liftovered/ADNI_merged-merge.missnp --make-bed --out ./liftovered/ADNI_1_tmp
plink --bfile ./liftovered/ADNI_2_1 --exclude ./liftovered/ADNI_merged-merge.missnp --make-bed --out ./liftovered/ADNI_2_1_tmp
plink --bfile ./liftovered/ADNI_2_2 --exclude ./liftovered/ADNI_merged-merge.missnp --make-bed --out ./liftovered/ADNI_2_2_tmp
plink --bfile ./liftovered/ADNI_3_1 --exclude ./liftovered/ADNI_merged-merge.missnp --make-bed --out ./liftovered/ADNI_3_1_tmp
plink --bfile ./liftovered/ADNI_3_2 --exclude ./liftovered/ADNI_merged-merge.missnp --make-bed --out ./liftovered/ADNI_3_2_tmp

echo "./liftovered/ADNI_2_1_tmp" > ./liftovered/all_datasets.txt
echo "./liftovered/ADNI_2_2_tmp" >> ./liftovered/all_datasets.txt
echo "./liftovered/ADNI_3_1_tmp" >> ./liftovered/all_datasets.txt
echo "./liftovered/ADNI_3_2_tmp" >> ./liftovered/all_datasets.txt

plink --bfile ./liftovered/ADNI_1_tmp --merge-list ./liftovered/all_datasets.txt --make-bed --out ./liftovered/ADNI_merged

rm ./liftovered/ADNI_1_tmp.*
rm ./liftovered/ADNI_2_1_tmp.*
rm ./liftovered/ADNI_2_2_tmp.*
rm ./liftovered/ADNI_3_1_tmp.*
rm ./liftovered/ADNI_3_2_tmp.*
