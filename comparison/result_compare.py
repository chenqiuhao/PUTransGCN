import os
import pandas as pd

# plot_dir = [
#     "./PDformer_two_step_spy",
#     "./PDformer_two_step",
#     "./PDformer_pu_bagging",
#     "./PDformer_all",
# ]

start = 199
end = start + 1
fw = pd.ExcelWriter(rf"result_compare{start}.xlsx")

n = 0
for root, dirs, files in os.walk("./"):
    if (
        "scores" in dirs
        and "archive" not in root
        and "(" not in root
        and "bak" not in root
    ):
        file_name = list(
            filter(lambda x: (x.find(".xlsx") >= 0) & (x.find("~$") == -1), files)
        )[0]
        model_name = file_name[:-5]
        # model_name = file_name[start:end]
        file_path = os.path.join(root, file_name)
        file = pd.read_excel(file_path, index_col=0, sheet_name=None)
        sheet0 = file["fold0"][start:end]
        sheet1 = file["fold1"][start:end]
        sheet2 = file["fold2"][start:end]
        sheet3 = file["fold3"][start:end]
        sheet4 = file["fold4"][start:end]
        sheet = pd.concat((sheet0, sheet1, sheet2, sheet3, sheet4))
        sheet_mean = sheet.mean()
        sheet_std = sheet.std()
        sheet_comb = pd.concat([sheet, sheet_mean.to_frame().T], ignore_index=True)
        sheet_comb = pd.concat([sheet_comb, sheet_std.to_frame().T], ignore_index=True)

        sheet_comb.to_excel(fw, sheet_name=model_name)
        result = sheet_mean.round(3).astype(str) + "±" + sheet_std.round(3).astype(str)
        result = result.to_frame().T.rename(index={0: model_name})
        if n == 0:
            all_result = result
            n = 1
        else:
            all_result = pd.concat([all_result, result])
    elif root == "./iPiDi-PUL":
        file_names = list(
            filter(lambda x: (x.find(".xlsx") >= 0) & (x.find("~$") == -1), files)
        )
        for file_name in file_names:
            model_name = file_name[:-5]
            file_path = os.path.join(root, file_name)
            file = pd.read_excel(file_path, index_col=0, sheet_name=None)
            sheet = file["fold0"]
            sheet_mean = sheet.mean()
            sheet_std = sheet.std()
            sheet_comb = pd.concat([sheet, sheet_mean.to_frame().T], ignore_index=True)
            sheet_comb = pd.concat(
                [sheet_comb, sheet_std.to_frame().T], ignore_index=True
            )

            sheet_comb.to_excel(fw, sheet_name=model_name)
            result = (
                sheet_mean.round(3).astype(str) + "±" + sheet_std.round(3).astype(str)
            )
            result = result.to_frame().T.rename(index={0: model_name})
            if n == 0:
                all_result = result
                n = 1
            else:
                all_result = pd.concat([all_result, result])


all_result.to_excel(fw, sheet_name="all_result")
fw.close()
