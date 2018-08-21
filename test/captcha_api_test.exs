defmodule CaptchaApiTest do
  use ExUnit.Case
  doctest CaptchaApi

  defp async_call_mean(index, data_file_name, column_name, verbose) do
    Task.async(fn ->
      :poolboy.transaction(
        :worker,
        fn pid ->
          GenServer.call(
            pid,
            {:mean, index, data_file_name, column_name, verbose})
        end,
        @timeout
      )
    end)
  end

  defp await_and_inspect_mean(task) do
    task
    |> Task.await(@timeout)
    |> (fn ({:result, value}) ->
      case value do
        {:not_found, msg} ->
          IO.puts(msg)
        {:invalid_column, msg} ->
          IO.puts(msg)
        {:ok, {index, data_file_name, column_name, mean}} ->
          IO.puts("index: #{index},   df: #{data_file_name}  col: #{column_name}  mean: #{mean}")
      end
    end).()
  end

end
