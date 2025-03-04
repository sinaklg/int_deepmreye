def adapt_evaluation(participant_evaluation):
    import pandas as pd
    import numpy as np
    pred_y = participant_evaluation["pred_y"]
    pred_y_median = np.nanmedian(pred_y, axis=1)
    pred_uncertainty = abs(participant_evaluation["euc_pred"])
    pred_uncertainty_median = np.nanmedian(pred_uncertainty, axis=1)
    df_pred_median = pd.DataFrame(
        np.concatenate(
            (pred_y_median, pred_uncertainty_median[..., np.newaxis]), axis=1),
        columns=["X", "Y", "Uncertainty"],
    )
    # With subTR
    subtr_values = np.concatenate((pred_y, pred_uncertainty[..., np.newaxis]),
                                  axis=2)
    index = pd.MultiIndex.from_product(
        [range(subtr_values.shape[0]),
         range(subtr_values.shape[1])],
        names=["TR", "subTR"])
    df_pred_subtr = pd.DataFrame(subtr_values.reshape(-1,
                                                      subtr_values.shape[-1]),
                                 index=index,
                                 columns=["X", "Y", "pred_error"])

    return df_pred_median, df_pred_subtr

def euclidean_distance(eye_data, pred_x, pred_y): 
     import numpy as np
     eucl_dist = np.sqrt(
        (eye_data[:int(len(pred_x)), 0] - pred_x[:int(len(eye_data))])**2 + (eye_data[:int(len(pred_y)), 1] - pred_y[:int(len(eye_data))])**2)

     return eucl_dist


def chunk_and_median(eyetracking_data, sampling_rate=1000, chunk_duration=1.2):
    import numpy as np
    """
    Splits continuous eyetracking data into chunks of specified duration
    and computes the median for each chunk, ensuring no NaNs are returned.

    Parameters:
    - eyetracking_data: 1D NumPy array (continuous signal)
    - sampling_rate: int, samples per second (default: 1000 Hz)
    - chunk_duration: float, duration of each chunk in seconds (default: 1.2 s)

    Returns:
    - medians: 1D NumPy array with median values per chunk
    """
    # Remove NaNs from input
    eyetracking_data = np.nan_to_num(eyetracking_data, nan=0.0)

    chunk_size = int(sampling_rate * chunk_duration)
    num_chunks = len(eyetracking_data) // chunk_size

    medians = np.array([
        np.nanmedian(eyetracking_data[i * chunk_size: (i + 1) * chunk_size])  # Ensures NaN-safe median
        for i in range(num_chunks)
    ])

    return medians




