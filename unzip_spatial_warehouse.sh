# for dir in train test; do
#     for subdir in images depths; do
#         if [ -d "$dir/$subdir" ]; then
#             echo "Processing $dir/$subdir"
#             cd "$dir/$subdir"
#             tar -xzf chunk_*.tar.gz
#             # rm chunk_*.tar.gz
#             cd ../..
#         fi
#     done
# done

tar -xzf train/images/chunk_000.tar.gz -C train/images/
tar -xzf train/images/chunk_001.tar.gz -C train/images/
tar -xzf train/images/chunk_002.tar.gz -C train/images/
tar -xzf train/images/chunk_003.tar.gz -C train/images/
tar -xzf train/images/chunk_004.tar.gz -C train/images/
tar -xzf train/images/chunk_005.tar.gz -C train/images/
tar -xzf train/images/chunk_006.tar.gz -C train/images/
tar -xzf train/images/chunk_007.tar.gz -C train/images/
tar -xzf train/images/chunk_008.tar.gz -C train/images/
tar -xzf train/images/chunk_009.tar.gz -C train/images/
tar -xzf train/images/chunk_010.tar.gz -C train/images/
tar -xzf train/images/chunk_011.tar.gz -C train/images/
tar -xzf train/images/chunk_012.tar.gz -C train/images/
tar -xzf train/images/chunk_013.tar.gz -C train/images/
tar -xzf train/images/chunk_014.tar.gz -C train/images/
tar -xzf train/images/chunk_015.tar.gz -C train/images/
tar -xzf train/images/chunk_016.tar.gz -C train/images/
tar -xzf train/images/chunk_017.tar.gz -C train/images/
tar -xzf train/images/chunk_018.tar.gz -C train/images/
tar -xzf train/images/chunk_019.tar.gz -C train/images/
tar -xzf train/images/chunk_020.tar.gz -C train/images/
tar -xzf train/images/chunk_021.tar.gz -C train/images/

tar -xzf test/images/chunk_000.tar.gz -C test/images
tar -xzf test/images/chunk_001.tar.gz -C test/images
tar -xzf test/images/chunk_002.tar.gz -C test/images
tar -xzf test/images/chunk_003.tar.gz -C test/images
tar -xzf test/images/chunk_004.tar.gz -C test/images