defmodule CaptchaApi.PythonServer do
  use GenServer
  alias CaptchaApi.ElixirPython
  require Logger

  def start_link(args) do
    GenServer.start_link(__MODULE__, :ok, args)
  end

  def init(:ok) do
    python_session = ElixirPython.start()
    ElixirPython.call(python_session, :predict, :load_neural)
    {:ok, python_session}
  end

  def predict(captcha) do
    GenServer.call(CaptchaApi.PythonServer, {:captcha, captcha}, :infinity)
  end

  def handle_call({:captcha, captcha}, _, session) do
    result = ElixirPython.call(session, :predict, :predict, [captcha])
    {:reply, result, session}
  end

  def handle_info({:python, message}, session) do
    IO.puts("Received message from python: #{inspect message}")
    {:stop, :normal,  session}
  end

  def terminate(_reason, session) do
    ElixirPython.stop(session)
    :ok
  end

end
