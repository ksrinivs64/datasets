import tarfile


def read_tar(filepath):
    problems_files2code = {}
    problems2files = {}
    with tarfile.open(filepath, "r") as tar:
        for tarinfo in tar:
            arr = tarinfo.name.split('/')

            if arr[4] not in problems2files:
                problems2files[arr[4]] = []
            code = tar.extractfile(tarinfo).readlines()
            modified_code = []
            for l in code:
                s = l.decode()
                if s == '':
                    continue
                modified_code.append(s.strip())
            code = ' '.join(modified_code)
            print(code)
            problems2files[arr[4]].append(tarinfo.name)
            problems_files2code[tarinfo.name] = code

    return problems_files2code, problems2files

def main():
    read_tar('/Users/kavithasrinivas/ibmcode/ai-for-code/datasets/CodeNet/java.test.tar')

if __name__ == "__main__":
    main()