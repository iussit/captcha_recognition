defmodule CaptchaApi.ElixirPython do
  @python_dir Path.expand("lib/python")

  def start() do
    {:ok, pid} = :python.start([{:python_path, to_charlist(@python_dir)}, {:python, 'python'}])
    pid
  end

  def call(pid, m, f, a \\ []) do
    :python.call(pid, m, f, a)
  end

  def cast(pid, message) do
    :python.cast(pid, message)
  end

  def stop(pid) do
    :python.stop(pid)
  end
end
