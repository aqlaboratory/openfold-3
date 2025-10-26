source /etc/os-release
case "$VERSION_ID" in
  7*)  repo_os=rhel7 ;;   # CentOS 7 base
  8*)  repo_os=rhel8 ;;   # Alma/RHEL 8 base
  *)   echo "Unsupported base image"; exit 1 ;;
esac

curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${repo_os}/x86_64/cuda-${repo_os}.repo" \
     -o /etc/yum.repos.d/cuda.repo
rpm --import "https://developer.download.nvidia.com/compute/cuda/repos/${repo_os}/x86_64/7fa2af80.pub"

yum install --setopt=install_weak_deps=False --setopt=tsflags=nodocs -y \
    cuda-minimal-build-12-6 \
    libcurand-devel-12-6 \
    libcublas-devel-12-6 \
    libcusparse-devel-12-6 \
    libcusolver-devel-12-6 \
    ninja-build 

echo "/usr/local/cuda-12.6/lib64" > /etc/ld.so.conf.d/cuda.conf && ldconfig
ln -s cuda-12.6 /usr/local/cuda

export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
